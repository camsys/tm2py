"""
"""

from collections import defaultdict as _defaultdict
from contextlib import contextmanager as _context
import os
from typing import Dict, Any, Tuple, Union

from tm2py.core.component import Component as _Component, Controller as _Controller
import tm2py.core.emme as _emme_tools
from tm2py.core.logging import LogStartEnd
from tm2py.core.tools import SpatialGridIndex

_crs_wkt = '''PROJCS["NAD83(HARN) / California zone 6 (ftUS)",GEOGCS["NAD83(HARN)",
DATUM["NAD83_High_Accuracy_Reference_Network",SPHEROID["GRS 1980",6378137,298.257222101,AUTHORITY["EPSG","7019"]],
TOWGS84[0,0,0,0,0,0,0],AUTHORITY["EPSG","6152"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",
0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4152"]],PROJECTION["Lambert_Conformal_Conic_2SP"],
PARAMETER["standard_parallel_1",33.88333333333333],PARAMETER["standard_parallel_2",32.78333333333333],
PARAMETER["latitude_of_origin",32.16666666666666],PARAMETER["central_meridian",-116.25],PARAMETER["false_easting",
6561666.667],PARAMETER["false_northing",1640416.667],UNIT["US survey foot",0.3048006096012192,AUTHORITY["EPSG",
"9003"]],AXIS["X",EAST],AXIS["Y",NORTH],AUTHORITY["EPSG","2875"]] '''


class CreateTODScenarios(_Component):
    """Highway assignment and skims"""

    def __init__(self, controller: _Controller):
        """Highway assignment and skims.

        Args:
            controller: parent Controller object
        """
        super().__init__(controller)
        self._emme_manager = None
        self._ref_auto_network = None

    def run(self):
        project_path = os.path.join(self.root_dir, self.config.emme.project_path)
        self._emme_manager = _emme_tools.EmmeManager()
        emme_app = self._emme_manager.project(project_path)
        self._emme_manager.init_modeller(emme_app)
        with self._setup():
            self._create_highway_scenarios()
            self._create_transit_scenarios()

    @_context
    def _setup(self):
        self._ref_auto_network = None
        try:
            yield
        finally:
            self._ref_auto_network = None

    def _project_coordinates(self, ref_scenario):
        modeller = self._emme_manager.modeller
        project_coord = modeller.tool(
            "inro.emme.data.network.base.project_network_coordinates")

        project_path = os.path.join(self.root_dir, self.config.emme.project_path)
        project_root = os.path.dirname(project_path)
        emme_app = self._emme_manager.project(project_path)
        src_prj_file = emme_app.project.spatial_reference_file
        if not src_prj_file:
            raise Exception(
                "Emme network coordinate reference system is not specified, unable to project coordinates for "
                "area type calculation. Set correct Spatial Reference in Emme Project settings -> GIS."
            )
        with open(src_prj_file, 'r') as src_prj:
            current_wkt = src_prj.read()
        if current_wkt != _crs_wkt:
            dst_prj_file = os.path.join(project_root, "Media", "NAD83(HARN) California zone 6 (ftUS).prj")
            with open(dst_prj_file, 'w') as dst_prj:
                dst_prj.write(_crs_wkt)
            project_coord(from_scenario=ref_scenario,
                          from_proj_file=src_prj_file,
                          to_proj_file=dst_prj_file,
                          overwrite=True)
            emme_app.project.spatial_reference.file_path = dst_prj_file
            emme_app.project.save()

    @LogStartEnd("Create highway time of day scenarios.")
    def _create_highway_scenarios(self):
        emmebank_path = os.path.join(self.root_dir, self.config.emme.highway_database_path)
        emmebank = self._emme_manager.emmebank(emmebank_path)
        ref_scenario = emmebank.scenario(self.config.emme.all_day_scenario_id)
        self._ref_auto_network = ref_scenario.get_network()
        self._emme_manager.change_emmebank_dimensions(
            emmebank,
            {"full_matrices": 9999, "extra_attribute_values": 40000000})
        # create VDFs & set cross-reference function parameters
        emmebank.extra_function_parameters.el1 = "@free_flow_time"
        emmebank.extra_function_parameters.el2 = "@capacity"
        emmebank.extra_function_parameters.el3 = "@ja"
        # TODO: should have just 3 functions, and map the FT to the vdf
        # TODO: could optimize expression (to review)
        bpr_tmplt = "el1 * (1 + 0.20 * ((volau + volad)/el2/0.75))^6"
        # "el1 * (1 + 0.20 * put(put((volau + volad)/el2/0.75))*get(1))*get(2)*get(2)"
        fixed_tmplt = "el1"
        akcelik_tmplt = (
            "(el1 + 60 * (0.25 *((((volau + volad)/el2) - 1) + "
            "((((((volau + volad)/el2) - 1)^2) + (16 * el3 * ("
            "(volau + volad)/el2)))^0.5))))"

            # "(el1 + 60 * (0.25 *(put(put((volau + volad)/el2) - 1) + "
            # "(((get(2)*get(2) + (16 * el3 * get(1)^0.5))))"
        )
        for f_id in ["fd1", "fd2", "fd9"]:
            if emmebank.function(f_id):
                emmebank.delete_function(f_id)
            emmebank.create_function(f_id, bpr_tmplt)
        for f_id in ["fd3", "fd4", "fd5", "fd7", "fd8", "fd10", "fd11", "fd12", "fd13", "fd14"]:
            if emmebank.function(f_id):
                emmebank.delete_function(f_id)
            emmebank.create_function(f_id, akcelik_tmplt)
        if emmebank.function("fd6"):
            emmebank.delete_function("fd6")
        emmebank.create_function("fd6", fixed_tmplt)

        ref_scenario = emmebank.scenario(self.config.emme.all_day_scenario_id)
        attributes = {
            "LINK": ["@area_type", "@capclass", "@free_flow_speed", "@free_flow_time"]
        }
        for domain, attrs in attributes.items():
            for name in attrs:
                if ref_scenario.extra_attribute(name) is None:
                    ref_scenario.create_extra_attribute(domain, name)

        network = ref_scenario.get_network()
        self._set_area_type(network)
        self._set_capclass(network)
        self._set_speed(network)
        ref_scenario.publish_network(network)
        self._ref_auto_network = network

        self._prepare_scenarios_and_attributes(emmebank)

    @LogStartEnd("Create transit time of day scenarios.")
    def _create_transit_scenarios(self):
        with self.logger.log_start_end("prepare base scenario"):
            emmebank_path = os.path.join(self.root_dir, self.config.emme.transit_database_path)
            emmebank = self._emme_manager.emmebank(emmebank_path)
            required_dims = {
                "full_matrices": 9999,
                "scenarios": 6,
                "regular_nodes": 550000,
                "links": 1200000,
                "transit_vehicles": 200,
                "extra_attribute_values": 100000000
            }
            self._emme_manager.change_emmebank_dimensions(emmebank, required_dims)
            for ident in ["ft1", "ft2", "ft3"]:
                if emmebank.function(ident):
                    emmebank.delete_function(ident)
            # for zero-cost links
            emmebank.create_function("ft1", "0")
            # segment travel time pre-calculated and stored in data1 (copied from @trantime_seg)
            emmebank.create_function("ft2", "us1")

            ref_scenario = emmebank.scenario(self.config.emme.all_day_scenario_id)
            attributes = {
                "LINK": ["@trantime", "@area_type", "@capclass", "@free_flow_speed", "@free_flow_time"]
            }
            for domain, attrs in attributes.items():
                for name in attrs:
                    if ref_scenario.extra_attribute(name) is None:
                        ref_scenario.create_extra_attribute(domain, name)
            network = ref_scenario.get_network()
            auto_network = self._ref_auto_network
            # copy link attributes from auto network to transit network
            link_lookup = {}
            for link in auto_network.links():
                link_lookup[link["#link_id"]] = link
            for link in network.links():
                auto_link = link_lookup.get(link["#link_id"])
                if not auto_link:
                    continue
                for attr in ["@area_type", "@capclass", "@free_flow_speed", "@free_flow_time"]:
                    link[attr] = auto_link[attr]

            mode_table = self.config.transit.modes
            walk_modes = set()
            access_modes = set()
            egress_modes = set()
            for mode_data in mode_table:
                mode = network.mode(mode_data['id'])
                if mode is None:
                    mode = network.create_mode(mode_data['assign_type'], mode_data['id'])
                elif mode.type != mode_data['assign_type']:
                    raise Exception(
                        f"mode {mode_data['id']} already exists with type {mode.type} instead of {mode_data['assign_type']}")
                mode.description = mode_data['name']
                if mode_data['assign_type'] == "AUX_TRANSIT":
                    # NOTE: Joel's version was using ul2*1.0 for spdfac,
                    #       only the TAP-TAP walk links had real distance, the others had distance of 999
                    mode.speed = mode_data['speed_miles_per_hour']
                if mode_data["type"] == "WALK":
                    walk_modes.add(mode.id)
                if mode_data["type"] == "ACCESS":
                    access_modes.add(mode.id)
                if mode_data["type"] == "EGRESS":
                    egress_modes.add(mode.id)
            aux_transit_modes = walk_modes | access_modes | egress_modes
            # TODO: validate at least one mode of each type
            # create vehicles
            vehicle_table = self.config.transit.vehicles
            for veh_data in vehicle_table:
                vehicle = network.transit_vehicle(veh_data['id'])
                if vehicle is None:
                    vehicle = network.create_transit_vehicle(veh_data['id'], veh_data['mode'])
                elif vehicle.mode.id != veh_data['mode']:
                    raise Exception(
                        f"vehicle {veh_data['id']} already exists with mode {vehicle.mode.id} instead of {veh_data['mode']}")
                vehicle.auto_equivalent = veh_data["auto_equivalent"]
                vehicle.seated_capacity = veh_data["seated_capacity"]
                vehicle.total_capacity = veh_data["total_capacity"]

            # set fixed guideway times, and initial free flow auto link times
            # TODO: to config
            cntype_speed_map = {"CRAIL": 45.0, "HRAIL": 40.0, "LRAIL": 30.0, "FERRY": 15.0}
            for link in network.links():
                speed = cntype_speed_map.get(link["#cntype"])
                if speed is None:
                    speed = link["@free_flow_speed"]
                    if link["@ft"] == 1 and speed > 0:
                        link["@trantime"] = 60 * link.length / speed
                    elif speed > 0:
                        link["@trantime"] = 60 * link.length / speed + link.length * 5 * 0.33
                else:
                    link["@trantime"] = 60 * link.length / speed
                # set TAP connector distance to 60 feet
                if link.i_node.is_centroid or link.j_node.is_centroid:
                    link.length = 0.01  # 60.0 / 5280.0
            for line in network.transit_lines():
                # TODO: may want to set transit line speeds (not necessarily used in the assignment though)
                line_veh = network.transit_vehicle(line["#mode"])
                if line_veh is None:
                    raise Exception(f"line {line.id} requires vehicle ('#mode') {line['#mode']} which does not exist")
                line_mode = line_veh.mode.id
                for seg in line.segments():
                    seg.link.modes |= {line_mode}
                line.vehicle = line_veh

            for link in network.links():
                # add access, egress and walk modes to links
                link.modes -= aux_transit_modes
                if link.i_node.is_centroid:
                    link.modes |= egress_modes
                elif link.j_node.is_centroid:
                    link.modes |= access_modes
                elif link["@walk_link"]:
                    link.modes |= walk_modes
                # TODO: remove transit modes which should not be available

            ref_scenario.publish_network(network)

        self._prepare_scenarios_and_attributes(emmebank)

        with self.logger.log_start_end("remove transit lines from other periods"):
            for period in self.config.periods:
                period_name = period.name.upper()
                with self.logger.log_start_end(f"period {period_name}"):
                    scenario = emmebank.scenario(period.emme_scenario_id)
                    network = scenario.get_network()
                    # removed transit lines from other periods from per-period scenarios
                    for line in network.transit_lines():
                        if line["#time_period"].upper() != period_name:
                            network.delete_transit_line(line)
                    scenario.publish_network(network)

    @LogStartEnd("Copy base to period scenarios and set per-period attributes")
    def _prepare_scenarios_and_attributes(self, emmebank):
        ref_scenario = emmebank.scenario(self.config.emme.all_day_scenario_id)
        # self._project_coordinates(ref_scenario)
        # find all time-of-day attributes (ends with period name)
        tod_attr_groups = {
            "NODE": _defaultdict(lambda: []),
            "LINK": _defaultdict(lambda: []), 
            "TURN": _defaultdict(lambda: []),
            "TRANSIT_LINE": _defaultdict(lambda: []), 
            "TRANSIT_SEGMENT": _defaultdict(lambda: []),
        }
        for attr in ref_scenario.extra_attributes():
            for period in self.config.periods:
                if attr.name.endswith(period.name):
                    tod_attr_groups[attr.type][attr.name[:-len(period.name)]].append(attr.name)
        for period in self.config.periods:
            scenario = emmebank.scenario(period.emme_scenario_id)
            if scenario:
                emmebank.delete_scenario(scenario)
            scenario = emmebank.copy_scenario(ref_scenario, period.emme_scenario_id)
            scenario.title = f"{period.name} {ref_scenario.title}"[:60]
            # in per-period scenario create attributes without period suffix, copy values 
            # for this period and delete all other period attributes
            for domain, all_attrs in tod_attr_groups.items():
                for root_attr, tod_attrs in all_attrs.items():
                    src_attr = f"{root_attr}{period.name}"
                    if root_attr.endswith("_"):
                        root_attr = root_attr[:-1]
                    for attr in tod_attrs:
                        if attr != src_attr:
                            scenario.delete_extra_attribute(attr)
                    attr = scenario.create_extra_attribute(domain, root_attr)
                    attr.description = scenario.extra_attribute(src_attr).description
                    values = scenario.get_attribute_values(domain, [src_attr])
                    scenario.set_attribute_values(domain, [root_attr], values)
                    scenario.delete_extra_attribute(src_attr)

    def _set_area_type(self, network):
        # set area type for links based on average density of MAZ closest to I or J node
        # the average density including all MAZs within the specified buffer distance
        buff_dist = 5280 * self.config.highway.area_type_buffer_dist_miles
        maz_data_file_path = os.path.join(self.root_dir, self.config.scenario.maz_landuse_file)
        maz_landuse_data: Dict[int, Dict[Any, Union[str, int, Tuple[float, float]]]] = {}
        with open(maz_data_file_path, 'r') as maz_data_file:
            header = [h.strip() for h in next(maz_data_file).split(",")]
            for line in maz_data_file:
                data = dict(zip(header, line.split(",")))
                maz_landuse_data[int(data["MAZ_ORIGINAL"])] = data
        # Build spatial index of MAZ node coords
        sp_index_maz = SpatialGridIndex(size=0.5 * 5280)
        for node in network.nodes():
            if node["@maz_id"]:
                x, y = node.x, node.y
                maz_landuse_data[int(node["@maz_id"])]["coords"] = (x, y)
                sp_index_maz.insert(int(node["@maz_id"]), x, y)
        for maz_landuse in maz_landuse_data.values():
            x, y = maz_landuse.get("coords", (None, None))
            if x is None:
                continue  # some MAZs in table might not be in network
            # Find all MAZs with the square buffer (including this one)
            # (note: square buffer instead of radius used to match earlier implementation)
            other_maz_ids = sp_index_maz.within_square(x, y, buff_dist)
            # Sum total landuse attributes within buffer distance
            total_pop = sum(int(maz_landuse_data[maz_id]["POP"]) for maz_id in other_maz_ids)
            total_emp = sum(int(maz_landuse_data[maz_id]["emp_total"]) for maz_id in other_maz_ids)
            total_acres = sum(float(maz_landuse_data[maz_id]["ACRES"]) for maz_id in other_maz_ids)
            # calculate buffer area type
            if total_acres > 0:
                density = (1 * total_pop + 2.5 * total_emp) / total_acres
            else:
                density = 0
            # code area type class
            if density < 6:
                maz_landuse["area_type"] = 5  # rural
            elif density < 30:
                maz_landuse["area_type"] = 4  # suburban
            elif density < 55:
                maz_landuse["area_type"] = 3  # urban
            elif density < 100:
                maz_landuse["area_type"] = 2  # urban business
            elif density < 300:
                maz_landuse["area_type"] = 1  # cbd
            else:
                maz_landuse["area_type"] = 0  # regional core
        # Find nearest MAZ for each link, take min area type of i or j node
        for link in network.links():
            i_node, j_node = link.i_node, link.j_node
            a_maz = sp_index_maz.nearest(i_node.x, i_node.y)
            b_maz = sp_index_maz.nearest(j_node.x, j_node.y)
            link["@area_type"] = min(
                maz_landuse_data[a_maz]["area_type"],
                maz_landuse_data[b_maz]["area_type"]
            )

    @staticmethod
    def _set_capclass(network):
        for link in network.links():
            area_type = link["@area_type"]
            if area_type < 0:
                link["@capclass"] = -1
            else:
                link["@capclass"] = 10 * area_type + link["@ft"]

    def _set_speed(self, network):
        free_flow_speed_map = {}
        for row in self.config.highway.capclass_lookup:
            if row.get("free_flow_speed") is not None:
                free_flow_speed_map[row["capclass"]] = row.get("free_flow_speed")
        for link in network.links():
            # default speed o 25 mph if missing or 0 in table map
            link["@free_flow_speed"] = free_flow_speed_map.get(link["@capclass"], 25)
            speed = link["@free_flow_speed"] or 25
            link["@free_flow_time"] = 60 * link.length / speed
