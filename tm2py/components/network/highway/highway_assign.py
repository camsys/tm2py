"""Highway assignment and skim component.

Performs equilibrium traffic assignment and generates resulting skims.
The assignmend is configured using the "highway" table in the source config.
See the config documentation for details. The traffic assignment runs according
to the list of assignment classes under highway.classes.

Other relevant parameters from the config are:
- emme.num_processors: number of processors as integer or "MAX" or "MAX-N"
- time_periods[].emme_scenario_id: Emme scenario number to use for each period
- time_periods[].highway_capacity_factor

The Emme network must have the following attributes available:

Link - attributes:
- "length" in feet
- "vdf", volume delay function (volume delay functions must also be setup)
- "@useclass", vehicle-class restrictions classification, auto-only, HOV only
- "@free_flow_time", the free flow time (in minutes)
- "@tollXX_YY", the toll for period XX and class subgroup (see truck
    class) named YY, used together with @tollbooth to generate @bridgetoll_YY
    and @valuetoll_YY
- "@maz_flow", the background traffic MAZ-to-MAZ SP assigned flow from highway_maz,
    if controller.iteration > 0
- modes: must be set on links and match the specified mode codes in
    the traffic config

 Network results - attributes:
- @flow_XX: link PCE flows per class, where XX is the class name in the config
- timau: auto travel time
- volau: total assigned flow in PCE

Notes:
- Output matrices are in miles, minutes, and cents (2010 dollars) and are stored/
as real values;
- Intrazonal distance/time is one half the distance/time to the nearest neighbor;
- Intrazonal bridge and value tolls are assumed to be zero
"""

from __future__ import annotations

import os
from contextlib import contextmanager as _context
from typing import TYPE_CHECKING, Dict, List, Union

import numpy as np
import pandas as pd

from tm2py import tools
from tm2py.components.component import Component
from tm2py.components.demand.prepare_demand import PrepareHighwayDemand
from tm2py.emme.manager import EmmeScenario
from tm2py.emme.matrix import MatrixCache, OMXManager
from tm2py.emme.network import NetworkCalculator
from tm2py.logger import LogStartEnd
import tm2py.emme as _emme_tools
from tm2py.components.network.highway.highway_network import PrepareNetwork

if TYPE_CHECKING:
    from tm2py.controller import RunController

    EmmeHighwayAnalysisSpec = Dict[
        str,
        Union[
            str,
            bool,
            None,
            Dict[
                str,
                Union[str, bool, None, Dict[str, Union[str, bool, None]]],
            ],
        ],
    ]
    EmmeHighwayClassSpec = Dict[
        str,
        Union[
            str,
            Dict[str, Union[str, float, Dict[str, str]]],
            List[EmmeHighwayAnalysisSpec],
        ],
    ]
    EmmeTrafficAssignmentSpec = Dict[
        str,
        Union[str, Union[str, bool, None, float, List[EmmeHighwayClassSpec]]],
    ]


class HighwayAssignment(Component):
    """Highway assignment and skims.

    Args:
        controller: parent RunController object
    """

    def __init__(self, controller: RunController):
        """Constructor for HighwayAssignment components.

        Args:
            controller (RunController): Reference to current run controller.
        """
        super().__init__(controller)

        self.config = self.controller.config.highway

        self._matrix_cache = None
        self._skim_matrices = []
        self._class_config = None

    @property
    def classes(self):
        # self.hwy_classes
        return [c.name for c in self.config.classes]

    @property
    def class_config(self):
        # self.hwy_class_configs
        if not self._class_config:
            self._class_config = {c.name: c for c in self.config.classes}

        return self._class_config

    def validate_inputs(self):
        """Validate inputs files are correct, raise if an error is found."""
        # TODO
        pass

    @LogStartEnd("Highway assignment and skims", level="STATUS")
    def run(self):
        """Run highway assignment."""
        demand = PrepareHighwayDemand(self.controller)
        iteration = self.controller.iteration
        dst_veh_groups = self.config.tolls.dst_vehicle_group_names
        run_dynamic_toll = self.config.tolls.run_dynamic_toll
        max_dynamic_toll_iter = self.config.tolls.max_dynamic_toll_iter
        valuetoll_start_tollbooth_code = self.config.tolls.valuetoll_start_tollbooth_code
        max_dynamic_valuetoll = self.config.tolls.max_dynamic_valuetoll
        warmstart = self.controller.config.run.warmstart.warmstart

        demand.run()
        for time in self.time_period_names:
            scenario = self.get_emme_scenario(
                self.controller.config.emme.highway_database_path, time
            )
            with self._setup(scenario, time):
                assign_classes = [
                    AssignmentClass(c, time, iteration, self.controller.config.run.warmstart.warmstart) for c in self.config.classes
                ]
                if (iteration > 0) & (self.config.run_maz_assignment):
                    self._copy_maz_flow(scenario)
                elif iteration == 0:
                    self._reset_background_traffic(scenario)
                else:
                    None
                self._create_skim_matrices(scenario, assign_classes)

                # pre-check if there are any valuetoll links
                valuetoll_start_tollbooth_code = self.config.tolls.valuetoll_start_tollbooth_code
                valuetoll_links_exist = False
                network = scenario.get_network()
                for link in network.links():
                    if link["@tollbooth"] >= valuetoll_start_tollbooth_code and link["@lanes"] > 0:
                        valuetoll_links_exist = True
                        break

                if (not run_dynamic_toll) or (not valuetoll_links_exist):
                    assign_spec = self._get_assignment_spec(assign_classes)
                    self._run_sola_traffic_assignment(scenario, assign_spec, chart_log_interval=1)
                elif iteration == 0 and (not warmstart):
                    # if run_dynamic_toll = True, warmstart = False, iteration = 0
                    assign_spec = self._get_assignment_spec(assign_classes)
                    self._run_sola_traffic_assignment(scenario, assign_spec, chart_log_interval=1)
                else: 
                    # (1) valuetoll_links_exist, run_dynamic_toll = True, warmstart = False, iteration >= 1, or
                    # (2) valuetoll_links_exist, run_dynamic_toll = True, warmstart = True
                    # run maximum "max_dynamic_toll_iter" times of dynamic tolling
                    # break out the loop if no valuetoll need to be updated

                    for dynamic_toll_iteration in range(1, max_dynamic_toll_iter+1):
                        assign_spec = self._get_assignment_spec(assign_classes)
                        # reduce inner iter if it's not the last dynamic toll iteration
                        if dynamic_toll_iteration < max_dynamic_toll_iter:
                            assign_spec["stopping_criteria"]["max_iterations"] = self.config.tolls.dynamic_toll_inner_iter
                        self._run_sola_traffic_assignment(scenario, assign_spec, chart_log_interval=1)
                        self._calc_total_flow(scenario)
                        self._calc_vc(scenario)

                        # reset indicator variable and attribute
                        update_toll_required = False
                        net_calc = NetworkCalculator(scenario)
                        net_calc("@update_dynamic_toll", "0")

                        # check if valuetoll need to be updated
                        network = scenario.get_network()
                        for link in network.links():
                            if link["@vc"] > 1 and link["@tollbooth"] >= valuetoll_start_tollbooth_code:
                                for dst_veh in dst_veh_groups:
                                    if link[f"@valuetoll_{dst_veh}"] < max_dynamic_valuetoll:
                                        # if any of the valuetoll field meet update criteria, flag the link
                                        update_toll_required = True
                                        link["@update_dynamic_toll"] = 1
                                        break 

                        if dynamic_toll_iteration < max_dynamic_toll_iter:
                            if update_toll_required: # update value tolls
                                PrepareNetwork(controller=self._controller)._set_dynamic_tolls(network=network)
                                PrepareNetwork(controller=self._controller)._calc_link_class_costs(network=network)
                                scenario.publish_network(network)
                            else:
                                # if there is no valuetoll updated needed before max_dynamic_toll_iter,
                                # run a full assignment without reducing inner iter
                                assign_spec = self._get_assignment_spec(assign_classes)
                                self._run_sola_traffic_assignment(scenario, assign_spec, chart_log_interval=1)
                                break # stop running another dynamic toll assignment iteration

                # after assignment (potentially with multiple iteration due to dynamic tolling) finished
                # Subtract non-time costs from gen cost to get the raw travel time
                for emme_class_spec in assign_spec["classes"]:
                    self._calc_time_skim(emme_class_spec)
                # Set intra-zonal for time and dist to be 1/2 nearest neighbour
                for class_config in self.config.classes:
                    self._set_intrazonal_values(
                        time,
                        class_config["name"],
                        class_config["skims"],
                    )
                self._export_skims(scenario, time)

                if self.logger.debug_enabled:
                    self._log_debug_report(scenario, time)

        # write result dynamic valuetoll to csv for reference
        if iteration == self.controller.config.run.end_iteration:
            self._write_dynamic_valuetoll()


    def _run_sola_traffic_assignment(self, scenario, assign_spec, chart_log_interval=1):
        with self.logger.log_start_end(
            "Run SOLA assignment with path analyses", level="INFO"
        ):
            assign = self.controller.emme_manager.tool(
                "inro.emme.traffic_assignment.sola_traffic_assignment"
            )
            assign(assign_spec, scenario, chart_log_interval)


    def _write_dynamic_valuetoll(self):
        iteration = self.controller.iteration
        output_valuetoll_path = self.get_abs_path(self.config.tolls.output_valuetoll_path)
        valuetoll_start_tollbooth_code = self.config.tolls.valuetoll_start_tollbooth_code
        dst_veh_groups = self.config.tolls.dst_vehicle_group_names

        if not os.path.exists(output_valuetoll_path): # make output_valuetoll_path folder if not exist
            os.makedirs(output_valuetoll_path)

        result_dynamic_valuetoll = pd.DataFrame()
        for time in self.time_period_names:
            scenario = self.get_emme_scenario(
                self.controller.config.emme.highway_database_path, time
            )
            network = scenario.get_network()
            period_valuetoll = {"tollbooth": [], "tollseg": [], "useclass": [], "toll_colname": [], "toll_val": []}
            for link in network.links():
                if link["@tollbooth"] >= valuetoll_start_tollbooth_code:
                    for dst_veh in dst_veh_groups:
                        period_valuetoll["tollbooth"].append(link["@tollbooth"])
                        period_valuetoll["tollseg"].append(link["@tollseg"])
                        period_valuetoll["useclass"].append(link["@useclass"])
                        period_valuetoll["toll_colname"].append(f"toll{time.lower()}_{dst_veh}")
                        valuetoll_per_mile = (link[f"@valuetoll_{dst_veh}"] / link.length) / 100  # calculate per-mile charge
                        period_valuetoll[f"toll_val"].append(valuetoll_per_mile)
            period_valuetoll = pd.DataFrame(period_valuetoll)
            period_valuetoll["fac_index"] = period_valuetoll["tollbooth"]*1000 + period_valuetoll["tollseg"]*10 + period_valuetoll["useclass"]
            period_valuetoll["tolltype"] = "expr_lane"
            period_valuetoll = period_valuetoll[["fac_index", "tollbooth", "tollseg", "tolltype", "useclass", "toll_colname", "toll_val"]]
            result_dynamic_valuetoll = pd.concat([result_dynamic_valuetoll, period_valuetoll]).reset_index(drop=True)

        try:
            result_dynamic_valuetoll = result_dynamic_valuetoll.drop_duplicates(subset=["fac_index", "tollbooth", "tollseg", "tolltype", "useclass", "toll_colname"])
            result_dynamic_valuetoll = pd.pivot(result_dynamic_valuetoll, index=["fac_index", "tollbooth", "tollseg", "tolltype", "useclass"], columns="toll_colname", values="toll_val").reset_index()
        except:
            self.logger.warn(f"toll values not unique for indexes, write out result dynamic valuetoll in long format", indent=True)
        result_dynamic_valuetoll.to_csv(f"{output_valuetoll_path}/result_dynamic_valuetolls.csv", index=False)


    @_context
    def _setup(self, scenario: EmmeScenario, time_period: str):
        """Setup and teardown for Emme Matrix cache and list of skim matrices.

        Args:
            scenario: Emme scenario object
            time_period: time period name
        """
        self._matrix_cache = MatrixCache(scenario)
        self._skim_matrices = []
        msg = f"Highway assignment for period {time_period}"
        with self.logger.log_start_end(msg, level="STATUS"):
            try:
                yield
            finally:
                self._matrix_cache.clear()
                self._matrix_cache = None
                self._skim_matrices = []

    def _copy_maz_flow(self, scenario: EmmeScenario):
        """Copy maz_flow from MAZ demand assignment to ul1 for background traffic.

        Args:
            scenario: Emme scenario object
        """
        self.logger.log_time(
            "Copy @maz_flow to ul1 for background traffic", indent=True, level="DETAIL"
        )
        net_calc = NetworkCalculator(scenario)
        net_calc("ul1", "@maz_flow")

    def _reset_background_traffic(self, scenario: EmmeScenario):
        """Set ul1 for background traffic to 0 (no maz-maz flow).

        Args:
            scenario: Emme scenario object
        """
        self.logger.log_time(
            "Set ul1 to 0 for background traffic", indent=True, level="DETAIL"
        )
        net_calc = NetworkCalculator(scenario)
        net_calc("ul1", "0")

    def _calc_total_flow(self, scenario: EmmeScenario):
        """Calculate total traffic flow by summing up flow by classes

        Args:
            scenario: Emme scenario object
        """
        assign_class_flow_names = []
        for assign_class in self.config.classes:
            assign_class_flow_names.append(f"@flow_{assign_class.name.lower()}")
        expression = " + ".join(assign_class_flow_names)

        net_calc = NetworkCalculator(scenario)
        net_calc("@total_flow", expression)

    def _calc_vc(self, scenario: EmmeScenario):
        """Calculate V/C Ratio

        Args:
            scenario: Emme scenario object
        """
        net_calc = NetworkCalculator(scenario)
        # for links with @capacity == 0, such as ml that has 0 lane in that period, keep vc as 0
        net_calc(result="@vc", expression="@total_flow / @capacity", selections={"link": "not @capacity=0"})

    def _create_skim_matrices(
        self, scenario: EmmeScenario, assign_classes: List[AssignmentClass]
    ):
        """Create matrices to store skim results in Emme database.

        Also add the matrices to list of self._skim_matrices.

        Args:
            scenario: Emme scenario object
            assign_classes: list of AssignmentClass objects
        """
        create_matrix = self.controller.emme_manager.tool(
            "inro.emme.data.matrix.create_matrix"
        )

        with self.logger.log_start_end("Creating skim matrices", level="DETAIL"):
            for klass in assign_classes:
                for matrix_name in klass.skim_matrices:
                    matrix = scenario.emmebank.matrix(f'mf"{matrix_name}"')
                    if not matrix:
                        matrix = create_matrix(
                            "mf", matrix_name, scenario=scenario, overwrite=True
                        )
                        self.logger.debug(
                            f"Create matrix name: {matrix_name}, id: {matrix.id}"
                        )
                    self._skim_matrices.append(matrix)

    def _get_assignment_spec(
        self, assign_classes: List[AssignmentClass]
    ) -> EmmeTrafficAssignmentSpec:
        """Generate template Emme SOLA assignment specification.

        Args:
            assign_classes: list of AssignmentClass objects

        Returns
            Emme specification for SOLA traffic assignment

        """
        relative_gap = self.config.relative_gap
        max_iterations = self.config.max_iterations
        # NOTE: mazmazvol as background traffic in link.data1 ("ul1")
        base_spec = {
            "type": "SOLA_TRAFFIC_ASSIGNMENT",
            "background_traffic": {
                "link_component": "ul1",
                "turn_component": None,
                "add_transit_vehicles": False,
            },
            "classes": [klass.emme_highway_class_spec for klass in assign_classes],
            "stopping_criteria": {
                "max_iterations": max_iterations,
                "best_relative_gap": 0.0,
                "relative_gap": relative_gap,
                "normalized_gap": 0.0,
            },
            "performance_settings": {
                "number_of_processors": self.controller.num_processors
            },
        }
        return base_spec

    def _calc_time_skim(self, emme_class_spec: EmmeHighwayClassSpec):
        """Calculate the real time skim =gen_cost-per_fac*link_costs.

        Args:
            emme_class_spec: dictionary of the per-class spec sub-section from the
                Emme SOLA assignment spec, classes list
        """
        od_travel_times = emme_class_spec["results"]["od_travel_times"][
            "shortest_paths"
        ]
        # TODO: revisit this
        class_name = od_travel_times.replace("mfGCTIME", "")
        times = f'mfTIME{class_name}'
        if od_travel_times is not None:
            # Total link costs is always the first analysis
            cost = emme_class_spec["path_analyses"][0]["results"]["od_values"]
            factor = emme_class_spec["generalized_cost"]["perception_factor"]
            gencost_data = self._matrix_cache.get_data(od_travel_times)
            cost_data = self._matrix_cache.get_data(cost)
            time_data = gencost_data - (factor * cost_data)
            self._matrix_cache.set_data(times, time_data)

    def _set_intrazonal_values(
        self, time_period: str, class_name: str, skims: List[str]
    ):
        """Set the intrazonal values to 1/2 nearest neighbour for time and distance skims.

        Args:
            time_period: time period name (from config)
            class_name: highway class name (from config)
            skims: list of requested skims (from config)
        """
        # use distance matrix to find disconnected zones
        dist_matrix_name = self.config.output_skim_matrixname_tmpl.format(
            class_name=class_name.upper(),
            property_name="DIST", # TODO: might have better implementation here
        )
        dist_data = self._matrix_cache.get_data(dist_matrix_name)
        rowsum = dist_data.sum(axis=1).tolist()
        colsum = dist_data.sum(axis=0).tolist()

        disconnected_zones = {"skim_row_sum": None, "skim_col_sum": None}
        disconnected_zones["skim_row_sum"] = rowsum
        disconnected_zones["skim_col_sum"] = colsum
        disconnected_zones = pd.DataFrame(disconnected_zones)
        disconnected_zones["taz"] = disconnected_zones.index + 1
        disconnected_zones = disconnected_zones[(disconnected_zones["skim_row_sum"] == 0) | (disconnected_zones["skim_col_sum"] == 0)].reset_index(drop=True)
        disconnected_zones = disconnected_zones["taz"].to_list()
        disconnected_zone_index = [x-1 for x in disconnected_zones]
        self.logger.warn(f"disconnected zones: {disconnected_zones}", indent=True)

        for skim_name in skims:
            matrix_name = self.config.output_skim_matrixname_tmpl.format(
                class_name=class_name.upper(),
                property_name=skim_name.upper(),
            )
            self.logger.debug(f"Setting intrazonals to 0.5*min for {matrix_name}")
            data = self._matrix_cache.get_data(matrix_name)
            # NOTE: sets values for external zones as well
            if "dist" in skim_name or "time" in skim_name or "cost" in skim_name:
                for zone_index in range(data.shape[0]):
                    if zone_index in disconnected_zone_index:
                        # update disconnected zone skim values
                        data[zone_index] = 1000000
                        data[:, zone_index] = 1000000
                    else:
                        # update other intrazonal skim values
                        row_skim_values = data[zone_index].tolist()
                        min_row_skim_value = min([x for x in row_skim_values if x != 0])
                        data[zone_index, zone_index] = 0.5 * min_row_skim_value # replace intra-zonal value with min_value
            else: # for other skims
                np.fill_diagonal(data, np.inf)
                data[np.diag_indices_from(data)] = 0.5 * np.nanmin(data, 1)
            self._matrix_cache.set_data(matrix_name, data)

    def _export_skims(self, scenario: EmmeScenario, time_period: str):
        """Export skims to OMX files by period.

        Args:
            scenario: Emme scenario object
            time_period: time period name
        """
        # NOTE: skims in separate file by period
        omx_file_path = self.get_abs_path(
            self.config.output_skim_path
        )
        self.logger.debug(
            f"export {len(self._skim_matrices)} skim matrices to {omx_file_path}"
        )
        os.makedirs(os.path.dirname(omx_file_path), exist_ok=True)
        with OMXManager(
            os.path.join(omx_file_path, self.config.output_skim_filename_tmpl.format(time_period=time_period)), 
            "w", scenario, matrix_cache=self._matrix_cache
        ) as omx_file:
            omx_file.write_matrices(self._skim_matrices)

    def _log_debug_report(self, scenario: EmmeScenario, time_period: str):
        num_zones = len(scenario.zone_numbers)
        num_cells = num_zones * num_zones
        self.logger.debug(f"Highway skim summary for period {time_period}")
        self.logger.debug(
            f"Number of zones: {num_zones}. Number of O-D pairs: {num_cells}. "
            "Values outside -9999999, 9999999 are masked in summaries."
        )
        self.logger.debug(
            "name                            min       max      mean           sum"
        )
        for matrix in self._skim_matrices:
            values = self._matrix_cache.get_data(matrix)
            data = np.ma.masked_outside(values, -9999999, 9999999)
            stats = (
                f"{matrix.name:25} {data.min():9.4g} {data.max():9.4g} "
                f"{data.mean():9.4g} {data.sum(): 13.7g}"
            )
            self.logger.debug(stats)


class AssignmentClass:
    """Highway assignment class, represents data from config and conversion to Emme specs."""

    def __init__(self, class_config, time_period, iteration, warmstart):
        """Constructor of Highway Assignment class.

        Args:
            class_config (_type_): _description_
            time_period (_type_): _description_
            iteration (_type_): _description_
            warmstart (Boolean): warmstart switch
        """
        self.class_config = class_config
        self.time_period = time_period
        self.iteration = iteration
        self.warmstart = warmstart
        self.name = class_config["name"].lower()
        self.skims = class_config.get("skims", [])

    @property
    def emme_highway_class_spec(self) -> EmmeHighwayClassSpec:
        """Construct and return Emme traffic assignment class specification.

        Converted from input config (highway.classes), see Emme Help for
        SOLA traffic assignment for specification details.
        Adds time_period as part of demand and skim matrix names.

        Returns:
            A nested dictionary corresponding to the expected Emme traffic
            class specification used in the SOLA assignment.
        """
        if (self.iteration == 0) and not self.warmstart:
            demand_matrix = 'ms"zero"'
        else:
            demand_matrix = f'mf"{self.time_period}_{self.name}"'
        class_spec = {
            "mode": self.class_config.mode_code,
            "demand": demand_matrix,
            "generalized_cost": {
                "link_costs": f"@cost_{self.name.lower()}",  # cost in $0.01
                # $/hr -> min/$0.01
                "perception_factor": 0.6 / self.class_config.value_of_time,
            },
            "results": {
                "link_volumes": f"@flow_{self.name.lower()}",
                "od_travel_times": {
                    "shortest_paths": f"mfGCTIME{self.name.upper()}"
                },
            },
            "path_analyses": self.emme_class_analysis,
        }
        return class_spec

    @property
    def emme_class_analysis(self) -> List[EmmeHighwayAnalysisSpec]:
        """Construct and return a list of path analyses specs which generate the required skims.

        Returns:
            A list of nested dictionaries corresponding to the Emme path analysis
            (per-class) specification used in the SOLA assignment.
        """
        class_analysis = []
        if "time" in self.skims:
            class_analysis.append(
                self.emme_analysis_spec(
                    f"@cost_{self.name}".lower(),
                    f"mfCOST{self.name.upper()}",
                )
            )
        for skim_type in self.skims:
            if skim_type in ["time", "gctime", "cost"]:
                continue
            group = self.name
            matrix_name = self.class_config.output_skim_matrixname_tmpl.format(
                property_name=skim_type.upper(),
                class_name=group.upper()
            )
            class_analysis.append(
                self.emme_analysis_spec(
                    self.skim_analysis_link_attribute(skim_type, group),
                    matrix_name,
                )
            )
        return class_analysis

    @property
    def skim_matrices(self) -> List[str]:
        """Returns: List of skim matrix names for this class."""
        skim_matrices = []
        if "time" in self.skims:
            skim_matrices.extend(
                [
                    f"TIME{self.name.upper()}",
                    f"GCTIME{self.name.upper()}",
                    f"COST{self.name.upper()}",
                ]
            )
        for skim_type in self.skims:
            if skim_type in ["time", "gctime", "cost"]:
                continue
            matrix_name = self.class_config.output_skim_matrixname_tmpl.format(
                property_name=skim_type.upper(),
                class_name=self.name.upper()
            )
            skim_matrices.append(matrix_name)
        return skim_matrices

    @staticmethod
    def emme_analysis_spec(link_attr: str, matrix_name: str) -> EmmeHighwayAnalysisSpec:
        """Returns Emme highway class path analysis spec.

        See Emme Help for SOLA assignment for full specification details.
        Args:
            link_attr: input link attribute for which to sum values along the paths
            matrix_name: full matrix name to store the result of the path analysis

        Returns:
            The nested dictionary specification which will generate the skim
            of link attribute values.
        """
        analysis_spec = {
            "link_component": link_attr,
            "turn_component": None,
            "operator": "+",
            "selection_threshold": {"lower": None, "upper": None},
            "path_to_od_composition": {
                "considered_paths": "ALL",
                "multiply_path_proportions_by": {
                    "analyzed_demand": False,
                    "path_value": True,
                },
            },
            "results": {
                "od_values": matrix_name,
                "selected_link_volumes": None,
                "selected_turn_volumes": None,
            },
        }
        return analysis_spec

    @staticmethod
    def skim_analysis_link_attribute(skim: str, group: str) -> str:
        """Return the link attribute name for the specified skim type and group.

        Args:
            skim: name of skim requested, one of dist, hovdist, tolldist, freeflowtime,
                bridgetoll, or valuetoll
            group: subgroup name for the bridgetoll or valuetoll, corresponds to one of
                the names from config.highway.tolls.dst_vehicle_group_names
        Returns:
            A string of the link attribute name used in the analysis.
        """
        lookup = {
            "dist": "length",  # NOTE: length must be in miles
            "hovdist": "@hov_length",
            "tolldist": "@toll_length",
            "freeflowtime": "@free_flow_time",
            "btoll": f"@bridgetoll_{group}",
            "vtoll": f"@valuetoll_{group}",
            "btoll_vsm": "@bridgetoll_vsm",
            "btoll_sml": "@bridgetoll_sml",
            "btoll_med": "@bridgetoll_med",
            "btoll_lrg": "@bridgetoll_lrg",
        }
        return lookup[skim]
