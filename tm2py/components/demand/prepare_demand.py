"""Demand loading from OMX to Emme database."""

from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING, Dict, List, Union
import pathlib

import numpy as np
import pandas as pd
from collections import defaultdict

from tm2py.components.component import Component
from tm2py.emme.matrix import OMXManager
from tm2py.logger import LogStartEnd

if TYPE_CHECKING:
    from tm2py.controller import RunController


class PrepareDemand(Component, ABC):
    """Abstract base class to import and average demand."""

    def __init__(self, controller: RunController):
        """Constructor for PrepareDemand class.

        Args:
            controller (RunController): Run controller for the current run.
        """
        super().__init__(controller)
        self._emmebank = None

    def _read(self, path, name, num_zones, factor=None):
        with OMXManager(path, "r") as omx_file:
            demand = omx_file.read(name)
        if demand is None:
            raise ValueError(f'Demand {name} not found in {path}')
        if factor is not None:
            demand = factor * demand
        demand = self._redim_demand(demand, num_zones)
        return demand

    @staticmethod
    def _redim_demand(demand, num_zones):
        _shape = demand.shape
        if _shape < (num_zones, num_zones):
            demand = np.pad(
                demand, ((0, num_zones - _shape[0]), (0, num_zones - _shape[1]))
            )
        elif _shape > (num_zones, num_zones):
            ValueError(
                f"Provided demand matrix is larger ({_shape}) than the \
                specified number of zones: {num_zones}"
            )

        return demand

    # Disable too many arguments recommendation
    # pylint: disable=R0913
    def _save_demand(self, name, demand, scenario, description="", apply_msa=False):
        matrix = self._emmebank.matrix(f'mf"{name}"')
        msa_iteration = self.controller.iteration
        end_iteration = self.controller.config.run.end_iteration
        # iteration 0 & 1: do not average demand
        # iteration 2 ~ end_iteration-1, use average demand
        # iteration end_iteration: do not average demand
        if (not apply_msa) or (msa_iteration <= 1) or (msa_iteration == end_iteration):
            if not matrix:
                ident = self._emmebank.available_matrix_identifier("FULL")
                matrix = self._emmebank.create_matrix(ident)
                matrix.name = name
                matrix.description = description
        else:
            if not matrix:
                raise Exception(f"error averaging demand: matrix {name} does not exist")
            prev_demand = matrix.get_numpy_data(scenario.id)
            demand = prev_demand + (1.0 / msa_iteration) * (demand - prev_demand)

        matrix.set_numpy_data(demand, scenario.id)

    def _create_zero_matrix(self):
        zero_matrix = self._emmebank.matrix('ms"zero"')
        if zero_matrix is None:
            ident = self._emmebank.available_matrix_identifier("SCALAR")
            zero_matrix = self._emmebank.create_matrix(ident)
            zero_matrix.name = "zero"
            zero_matrix.description = "zero demand matrix"
        zero_matrix.data = 0


class PrepareHighwayDemand(PrepareDemand):
    """Import and average highway demand.

    Demand is imported from OMX files based on reference file paths and OMX
    matrix names in highway assignment config (highway.classes).
    The demand is average using MSA with the current demand matrices
    (in the Emmebank) if the controller.iteration > 1.

    Args:
        controller: parent RunController object
    """

    def __init__(self, controller: RunController):
        """Constructor for PrepareHighwayDemand.

        Args:
            controller (RunController): Reference to run controller object.
        """
        super().__init__(controller)
        self._emmebank_path = None

    def validate_inputs(self):
        # TODO
        pass

    # @LogStartEnd("prepare highway demand")
    def run(self):
        """Open combined demand OMX files from demand models and prepare for assignment."""
        self._emmebank_path = self.get_abs_path(self.controller.config.emme.highway_database_path)

        self._emmebank = self.controller.emme_manager.emmebank(self._emmebank_path)
        self._create_zero_matrix()
        for time in self.time_period_names:
            for klass in self.controller.config.highway.classes:
                self._prepare_demand(klass.name, klass.description, klass.demand, time)

    def _prepare_demand(
        self,
        name: str,
        description: str,
        demand_config: List[Dict[str, Union[str, float]]],
        time_period: str,
    ):
        """Load demand from OMX files and save to Emme matrix for highway assignment.

        Average with previous demand (MSA) if the current iteration > 1

        Args:
            name (str): the name of the highway assignment class
            description (str): the description for the highway assignment class
            demand_config (dict): the list of file cross-reference(s) for the demand to be loaded
                {"source": <name of demand model component>,
                 "name": <OMX key name>,
                 "factor": <factor to apply to demand in this file>}
            time_period (str): the time time_period ID (name)
        """
        apply_msa = self.controller.config.highway.msa.apply_msa
        scenario = self.get_emme_scenario(self._emmebank.path, time_period)
        num_zones = len(scenario.zone_numbers)
        demand = self._read_demand(demand_config[0], time_period, num_zones)
        for file_config in demand_config[1:]:
            demand = demand + self._read_demand(file_config, time_period, num_zones)
        demand_name = f"{time_period}_{name}"
        description = f"{time_period} {description} demand"
        self._save_demand(demand_name, demand, scenario, description, apply_msa=apply_msa)
        
    def export_trip_tables(self):
    
        """Load demand from OMX files and aggregate as daily demand for evaluating convergence."""
        
        triptables = OMXManager(
            self.get_abs_path(self.controller.config.highway.convergence.output_triptable_path.format(iteration = self.controller.iteration)),
            "w")
        
        triptables.open()
        
        num_total_zones = self.num_total_zones
        
        # Highway, passenger and truck
        for time_period in self.time_period_names:
            scenario = self.get_emme_scenario(self._emmebank.path, time_period)
            for klass in self.controller.config.highway.classes:
                name, description, demand_config = klass.name, klass.description, klass.demand
                demand = np.zeros((num_total_zones, num_total_zones))
                for file_config in demand_config:
                    demand += self._read_demand(file_config, time_period, num_total_zones)
                triptables.write_array(demand, name = f'{name}_{time_period}')
        
        # Transit
        for time_period in self.time_period_names:
            scenario = self.get_emme_scenario(self._emmebank.path, time_period)
            for klass in self.controller.config.transit.classes:
                name, description = klass.name.upper(), klass.description
                path = self.get_abs_path(self.controller.config.household.transit_demand_file)
                demand = self._read(path.format(period=time_period), name, num_total_zones)
                triptables.write_array(demand, name = f'{name}_{time_period}')
        
        triptables.close()

    def _read_demand(self, file_config, time_period, num_zones):
        # Load demand from cross-referenced source file,
        # the named demand model component under the key highway_demand_file
        source = file_config["source"]
        name = file_config["name"].format(period=time_period.upper())
        factor = file_config.get("factor")
        path = self.get_abs_path(self.controller.config[source].highway_demand_file)
        return self._read(path.format(period=time_period), name, num_zones, factor)
    
    @LogStartEnd("Prepare household demand matrices.")
    def prepare_household_demand(self):
        """Prepares highway and transit household demand matrices from trip lists produced by CT-RAMP.
        """
        iteration = self.controller.iteration
        
        # Create folders if they don't exist
        pathlib.Path(self.get_abs_path(self.controller.config.household.highway_demand_file)).parents[0].mkdir(parents=True, exist_ok=True) 
        pathlib.Path(self.get_abs_path(self.controller.config.household.transit_demand_file)).parents[0].mkdir(parents=True, exist_ok=True) 
        pathlib.Path(self.get_abs_path(self.controller.config.household.active_demand_file)).parents[0].mkdir(parents=True, exist_ok=True) 
        
        
        indiv_trip_file = pathlib.Path(self.controller.config.household.ctramp_run_dir) / self.controller.config.household.ctramp_indiv_trip_file.format(iteration = iteration)
        joint_trip_file = pathlib.Path(self.controller.config.household.ctramp_run_dir) / self.controller.config.household.ctramp_joint_trip_file.format(iteration = iteration)
        it_full, jt_full = pd.read_csv(indiv_trip_file), pd.read_csv(joint_trip_file)
        
        # Add time period, expanded count
        time_period_start = dict(zip(
            [c.name.upper() for c in self.controller.config.time_periods], 
            [c.start_hour for c in self.controller.config.time_periods]))
        # the last time period needs to be filled in because the first period may or may not start at midnight
        time_periods_sorted = sorted(time_period_start, key = lambda x:time_period_start[x]) # in upper case
        first_period = time_periods_sorted[0]
        periods_except_last = time_periods_sorted[:-1]
        breakpoints = [time_period_start[tp] for tp in time_periods_sorted]
        it_full['time_period'] = pd.cut(it_full.depart_hour, breakpoints, right = False, labels = periods_except_last).cat.add_categories(time_periods_sorted[-1]).fillna(time_periods_sorted[-1]).astype(str)
        jt_full['time_period'] = pd.cut(jt_full.depart_hour, breakpoints, right = False, labels = periods_except_last).cat.add_categories(time_periods_sorted[-1]).fillna(time_periods_sorted[-1]).astype(str)
        it_full['eq_cnt'] = 1/it_full.sampleRate
        jt_full['eq_cnt'] = jt_full.num_participants/jt_full.sampleRate
        zp_cav = self.controller.config.household.OwnedAV_ZPV_factor
        zp_tnc = self.controller.config.household.TNC_ZPV_factor

        num_zones = self.num_internal_zones
        OD_full_index = pd.MultiIndex.from_product([range(1,num_zones + 1), range(1,num_zones + 1)])
        
        def combine_trip_lists(it, jt, trip_mode):
            # combines individual trip list and joint trip list
            combined_trips = pd.concat([it[(it['trip_mode'] == trip_mode)], jt[(jt['trip_mode'] == trip_mode)]])
            combined_sum = combined_trips.groupby(['orig_taz','dest_taz'])['eq_cnt'].sum()
            return combined_sum.reindex(OD_full_index, fill_value=0).unstack().values
        
        def create_zero_passenger_trips(trips, deadheading_factor, trip_modes=[1,2,3]):
            zpv_trips = trips.loc[(trips['avAvailable']==1) & (trips['trip_mode'].isin(trip_modes))]
            zpv_trips['eq_cnt'] = zpv_trips['eq_cnt'] * deadheading_factor
            zpv_trips = zpv_trips.rename(columns={'dest_taz': 'orig_taz', 
                                        'orig_taz': 'dest_taz'})
            return zpv_trips

        # create zero passenger trips for auto modes
        it_zpav_trp = create_zero_passenger_trips(it_full, zp_cav, trip_modes=[1,2,3])
        jt_zpav_trp = create_zero_passenger_trips(jt_full, zp_cav, trip_modes=[1,2,3])

        # create zero passenger trips for TNC modes
        it_zptnc_trp = create_zero_passenger_trips(it_full, zp_tnc, trip_modes=[9])
        jt_zptnc_trp = create_zero_passenger_trips(jt_full, zp_tnc, trip_modes=[9])

        # Combining zero passenger trips to trip files
        it_full = pd.concat([it_full, it_zpav_trp, it_zptnc_trp], ignore_index=True).reset_index(drop=True)
        jt_full = pd.concat([jt_full, jt_zpav_trp, jt_zptnc_trp], ignore_index=True).reset_index(drop=True)

        # read properties from config
        
        mode_name_dict = self.controller.config.household.ctramp_mode_names
        income_segment_config = self.controller.config.household.income_segment
        
        if income_segment_config['enabled']:
            # This only affects highway trip tables.
            
            hh_file = pathlib.Path(self.controller.config.household.ctramp_run_dir) / self.controller.config.household.ctramp_hh_file.format(iteration = iteration)
            hh = pd.read_csv(hh_file, usecols = ['hh_id', 'income'])
            it_full = it_full.merge(hh, on = 'hh_id', how = 'left')
            jt_full = jt_full.merge(hh, on = 'hh_id', how = 'left')
            
            suffixes = income_segment_config['segment_suffixes']
            
            it_full['income_seg'] = pd.cut(it_full['income'], right =False, 
                               bins = income_segment_config['cutoffs'] + [float('inf')], 
                               labels = suffixes).astype(str)

            jt_full['income_seg'] = pd.cut(jt_full['income'], right =False, 
                               bins = income_segment_config['cutoffs'] + [float('inf')], 
                               labels = suffixes).astype(str)
        else: 
            it_full['income_seg'] = ''
            jt_full['income_seg'] = ''
            suffixes = ['']
        
        # groupby objects for combinations of time period - income segmentation, used for highway modes only
        it_grp = it_full.groupby(['time_period', 'income_seg'])
        jt_grp = jt_full.groupby(['time_period', 'income_seg'])
        
        for time_period in time_periods_sorted:
            self.logger.debug(f"Producing household demand matrices for period {time_period}")
            
            highway_out_file = OMXManager(
                self.get_abs_path(self.controller.config.household.highway_demand_file).format(period=time_period), 'w')
            transit_out_file = OMXManager(
                self.get_abs_path(self.controller.config.household.transit_demand_file).format(period=time_period), 'w')
            active_out_file = OMXManager(
                self.get_abs_path(self.controller.config.household.active_demand_file).format(period=time_period), 'w')

            highway_out_file.open()
            transit_out_file.open()
            active_out_file.open()
            
            
            # Transit and active modes: one matrix per time period per mode
            it = it_full[it_full.time_period == time_period]
            jt = jt_full[jt_full.time_period == time_period]
            
            for trip_mode in mode_name_dict:
                if trip_mode in [4,5]:
                    matrix_name =  mode_name_dict[trip_mode]
                    self.logger.debug(f"Writing out mode {mode_name_dict[trip_mode]}")
                    active_out_file.write_array(numpy_array=combine_trip_lists(it,jt, trip_mode), name = matrix_name)
                    
                elif trip_mode == 6:
                    matrix_name = "WLK_TRN_WLK"
                    self.logger.debug(f"Writing out mode WLK_TRN_WLK")
                    transit_out_file.write_array(numpy_array=combine_trip_lists(it,jt, trip_mode), name = matrix_name)
                    
                elif trip_mode in [7,8]:
                    it_outbound, it_inbound = it[it.inbound == 0], it[it.inbound == 1]
                    jt_outbound, jt_inbound = jt[jt.inbound == 0], jt[jt.inbound == 1]
                    
                    matrix_name = f'{mode_name_dict[trip_mode].upper()}_TRN_WLK' 
                    
                    self.logger.debug(f"Writing out mode {mode_name_dict[trip_mode].upper() + '_TRN_WLK'}")
                    transit_out_file.write_array(
                        numpy_array=combine_trip_lists(it_outbound,jt_outbound, trip_mode), 
                        name = matrix_name)
                    
                    matrix_name = f'WLK_TRN_{mode_name_dict[trip_mode].upper()}' 
                    
                    self.logger.debug(f"Writing out mode {'WLK_TRN_' + mode_name_dict[trip_mode].upper()}")
                    transit_out_file.write_array(
                        numpy_array=combine_trip_lists(it_inbound,jt_inbound, trip_mode), 
                        name = matrix_name)
            

            # Highway modes: one matrix per suffix (income class) per time period per mode
            for suffix in suffixes:

                highway_cache = {}
                
                if (time_period, suffix) in it_grp.groups.keys():
                    it = it_grp.get_group((time_period, suffix))
                else:
                    it = pd.DataFrame(None, columns = it_full.columns)
                    
                if (time_period, suffix) in jt_grp.groups.keys():
                    jt = jt_grp.get_group((time_period, suffix))
                else:
                    jt = pd.DataFrame(None, columns = jt_full.columns)

               
                for trip_mode in sorted(mode_name_dict):
                    # Python preserves keys in the order they are inserted but
                    # mode_name_dict originates from TOML, which does not guarantee
                    # that the ordering of keys is preserved.  See
                    # https://github.com/toml-lang/toml/issues/162

                    if trip_mode in [1,2,3]: # currently hard-coded based on Travel Mode trip mode codes
                        highway_cache[mode_name_dict[trip_mode]] = combine_trip_lists(it,jt, trip_mode)

                    elif trip_mode == 9:
                        # identify the correct mode split factors for da, sr2, sr3
                        self.logger.debug(f"Splitting ridehail trips into shared ride trips")
                        ridehail_split_factors = defaultdict(float)
                        splits = self.controller.config.household.rideshare_mode_split
                        for key in splits:
                            out_mode_split = self.controller.config.household.__dict__[f'{key}_split']
                            for out_mode in out_mode_split:
                                ridehail_split_factors[out_mode] += out_mode_split[out_mode] * splits[key]
                                
                        ridehail_trips = combine_trip_lists(it,jt, trip_mode)
                        ridehail_zombie_trips = create_zero_passenger_vehicle_trips(it, jt, trip_mode, self.controller.config.household.TNC_ZPV_factor)
                        for out_mode in ridehail_split_factors:
                            matrix_name =f'{out_mode}_{suffix}'  if suffix else out_mode
                            self.logger.debug(f"Writing out mode {out_mode}")
                            highway_cache[out_mode] += (ridehail_trips * ridehail_split_factors[out_mode]).astype(float).round(2)
                            highway_out_file.write_array(numpy_array = highway_cache[out_mode], name = matrix_name)
       
            highway_out_file.close()
            transit_out_file.close()
            active_out_file.close()
    
    @property
    def num_internal_zones(self):
        return len(pd.read_csv(
            self.get_abs_path(self.controller.config.scenario.landuse_file), usecols = [self.controller.config.scenario.landuse_index_column]))
        
    @property
    def num_total_zones(self):
        self._emmebank_path = self.get_abs_path(self.controller.config.emme.highway_database_path)
        self._emmebank = self.controller.emme_manager.emmebank(self._emmebank_path)
        time_period = self.controller.config.time_periods[0].name
        scenario = self.get_emme_scenario(self._emmebank.path, time_period) # any scenario id works 
        return len(scenario.zone_numbers)
        
# class PrepareTransitDemand(PrepareDemand):
#     """Import transit demand."""
#
#     def run(self, time_period: Union[Collection[str], str] = None):
#         """Open combined demand OMX files from demand models and prepare for assignment.
#
#         Args:
#             time_period: list of str names of time_periods, or name of a single time_period
#         """
#         emmebank_path = self.get_abs_path(self.config.emme.transit_database_path)
#         self._emmebank = self.controller.emme_manager.emmebank(emmebank_path)
#         self._create_zero_matrix()
