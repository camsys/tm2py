"""Accessibility Components"""

import itertools, pathlib, os
from collections import defaultdict
from typing import TYPE_CHECKING, Collection, Mapping, Union

import pandas as pd
import numpy as np
from numpy import array as NumpyArray

from tm2py.components.component import Component
from tm2py.components.network.skims import get_omx_skim_as_numpy, get_summed_skims
from tm2py.components.network.postprocess_skims import HighwayPostprocessor, TransitPostprocessor
from tm2py.logger import LogStartEnd
from tm2py.emme.matrix import OMXManager

if TYPE_CHECKING:
    from tm2py.controller import RunController

class HomeAccessibility(Component):
    """Computes relative, unite-less measures of accessibility for home loctn as f(skims,land use).

    Used by the automobile ownership model. Uses Parameters from HomeAccessibilityConfig.

    Steps:
    1.  Multiplies an employment variable by a mode-specific decay function.  
    The product reflects the difficulty of accessing the activities the farther 
    (in terms of round-trip travel time) the jobs are from the location in  question. 
    
    2. The products to each destination zone are summed over each origin zone, and the 
    logarithm of the product mutes large differences.  The decay function on the walk 
    accessibility measure is steeper than automobile or transit.  
    
    The minimum accessibility is zero.  

    Inputs:
    (A) Highway skims for the AM peak period, midday period, and PM peak periods. 
    Each skim mode is expected to include the tables for the properties:
        (i) "TOLLTIMEDA": drive alone in-vehicle travel time for automobiles willing to pay a 
        "value" (time-savings) toll.  This path is used as a proxy for automobile travel time. 
    (B) Transit skims for the AM peak period, midday period, and PM peak periods.
    The skims are from the transit paths in which all line-haul modes are weighted equally.  
    Each skim is expected to include the following tables: 
        (i) "IVT", in-vehicle time; 
        (ii) "IWAIT", initial wait time; 
        (iii) "XWAIT",transfer wait time; 
        (iv) "WACC", walk access time; 
        (v) "WAUX", auxiliary walk time; and, 
        (vi) "WEGR", walk egress time.  
        
    
    (C) Zonal data file in which must include the following variables: 
        (i) "TOTEMP", total employment; 
        (ii) "RETEMPN", retail trade employment per the NAICS classification. 

    Outputs: CSV file with the following data items : 
    (i)    taz, travel analysis zone number; 
    (ii)   autoPeakRetail, the accessibility by automobile during peak conditions to retail employment for this TAZ; 
    (iii)  autoPeakTotal, the accessibility by automobile during peak conditions to all employment; 
    (iv)   autoOffPeakRetail, the accessibility by automobile during off-peak conditions to retail employment; 
    (v)    autoOffPeakTotal, the accessibility by automobile during off-peak conditions to all employment; 
    (vi)   transitPeakRetail, the accessibility by transit during peak conditions to retail employment; 
    (vii)  transitPeakTotal, the accessibility by transit during peak conditions to all employment;
    (viii) transitOffPeakRetail, the accessiblity by transit during off-peak conditions to retail employment;
    (ix)   transitOffPeakTotal, the accessiblity by transit during off-peak conditions to all employment;
    (x)    nonMotorizedRetail, the accessibility by walking during all time periods to retail employment;
    (xi)   nonMotorizedTotal, the accessibility by walking during all time periods to all employment. 

    Estimation:
    TODO

    Provenance:  
    eas (2022) dto (2010 08 30); gde (2009 04 30); bts (2009 04 09); 
    jh (2008 11 10); dto (2007 12 20)

    
    Properties:
        controller: parent RunController object
    """

    def __init__(self, controller):

        super().__init__(controller)
        
        self.config = self.controller.config.accessibility
        self.zonal_data_file = self.get_abs_path(
            self.controller.config.scenario.landuse_file
        )
        
        tree = lambda: defaultdict(tree)
        self._skims = tree()
        self._employment_df = None
        self.logsums_df = pd.DataFrame()

    @property
    def emmebank(self):
        """Reference to highway assignment Emmebank.

        TODO
            This should really be in the controller?
            Or part of network.skims?
        """
        return self.controller.emme_manager.emmebank(
            self.get_abs_path(self.controller.config.emme.highway_database_path)
        )
    
    @property
    def emme_scenario(self):
        """Return emme scenario from emmebank.

        Use first valid scenario for reference Zone IDs.

        TODO
            This should really be in the controller?
            Or part of network.skims?
        """
        _ref_scenario_id = self.controller.config.time_periods[0].emme_scenario_id
        if self.emmebank.scenario(_ref_scenario_id):
            return self.emmebank.scenario(_ref_scenario_id)
        return self.emmebank.scenario(1)

    @property
    def employment_df(self):
        if self._employment_df is None:
            self._employment_df = self.get_employment_df()
        return self._employment_df
    
    def get_summed_skims_from_config(self, mode, time_period, skim_prop=None):
        formulas = self.config.__dict__[f'formula_{mode}']
        if skim_prop is None:
            skim_prop = formulas['prop']
        if mode == 'auto':
            return np.add.reduce([get_omx_skim_as_numpy(self.controller, *matrix, property=skim_prop) for matrix in formulas[time_period][0]]) + \
                np.add.reduce([get_omx_skim_as_numpy(self.controller, *matrix, property=skim_prop) for matrix in formulas[time_period][1]]).T
        elif mode == 'transit':
            ivt_matrix_names = [dict(zip(['skim_mode','time_period','property'], matrix_name[0] + [matrix_name[1]])) for matrix_name in itertools.product(formulas[time_period][0], formulas['ivt'])]
            ivt_matrix_names_T = [dict(zip(['skim_mode','time_period','property'], matrix_name[0] + [matrix_name[1]])) for matrix_name in itertools.product(formulas[time_period][1], formulas['ivt'])]
            ovt_matrix_names = [dict(zip(['skim_mode','time_period','property'], matrix_name[0] + [matrix_name[1]])) for matrix_name in itertools.product(formulas[time_period][0], formulas['ovt'])]
            ovt_matrix_names_T = [dict(zip(['skim_mode','time_period','property'], matrix_name[0] + [matrix_name[1]])) for matrix_name in itertools.product(formulas[time_period][1], formulas['ovt'])]
           
            ivt = np.add.reduce([get_omx_skim_as_numpy(self.controller, **matrix) for matrix in ivt_matrix_names]) + np.add.reduce([get_omx_skim_as_numpy(self.controller, **matrix) for matrix in ivt_matrix_names_T]).T
            ovt =  np.add.reduce([get_omx_skim_as_numpy(self.controller, **matrix) for matrix in ovt_matrix_names]) + np.add.reduce([get_omx_skim_as_numpy(self.controller, **matrix) for matrix in ovt_matrix_names_T]).T
            return ivt, ovt
        elif mode == 'walk':
            return np.add.reduce([get_omx_skim_as_numpy(self.controller, *matrix, skim_prop) for matrix in formulas[time_period][0]]) + \
                np.add.reduce([get_omx_skim_as_numpy(self.controller, *matrix, skim_prop) for matrix in formulas[time_period][1]]).T
        
    
    def skims(self,mode,time_period,skim_prop=None):
        # TODO all the skim manipulations in https://github.com/BayAreaMetro/travel-model-one/blob/master/model-files/scripts/skims/Accessibility.job
        if len(self._skims[mode][time_period][skim_prop]) == 0:
            formulas = self.config.__dict__[f'formula_{mode}']
            skim_prop = formulas['prop']
            if mode == 'auto':
                if time_period == 'peak':

                    self._skims[mode][time_period][skim_prop] = self.get_summed_skims_from_config(mode, time_period, skim_prop)
                else:

                    self._skims[mode][time_period][skim_prop] = self.get_summed_skims_from_config(mode, time_period, skim_prop)

            elif mode == 'transit':
                #TODO need to sum up all the properties....
                self._skims[mode][time_period][skim_prop] = self.get_transit_skim(
                    mode,
                    time_period,
                    skim_prop
                )

            elif mode == 'walk':
                distance = self.get_summed_skims_from_config(mode, 'peak', skim_prop)

                self._skims[mode][time_period][skim_prop] = np.where(distance > self.config.__dict__['max_walk_distance'], 0, distance)
                

        return self._skims[mode][time_period][skim_prop]
        
    def get_transit_skim(self,mode:str,time_period:str,skim_prop:str):
        """ 
        inVehicleTime    = mi.6.IVT.T[j]
        outOfVehicleTime = @token_out_of_vehicle_time_weight@ * (mi.6.IWAIT.T[j] + mi.6.XWAIT.T[j] + mi.6.WACC.T[j] + mi.6.WAUX.T[j] + mi.6.WEGR.T[j])
        trPkTime_do     = (inVehicleTime + outOfVehicleTime)/100.0

        Args:
            mode (_type_): _description_
            time_period (_type_): _description_
            skim_prop (_type_): _description_
        """
        ivt, ovt = self.get_summed_skims_from_config(mode, time_period, skim_prop)

            
        return (ivt + self.config.__dict__['out_of_vehicle_time_weight'] * ovt)/100

    def get_employment_df(self):
        """Aggregates landuse data from input CSV by MAZ to TAZ and employment groups.

        TOTEMP, total employment (same regardless of classification system)
        RETEMPN, retail trade employment per the NAICS classification system
        """
        lu_maz_df = pd.read_csv(self.zonal_data_file)
        
        lu_maz_df = lu_maz_df[lu_maz_df["ZONE"].isin(self.zones)]
        lu_taz_df = lu_maz_df.groupby(["ZONE"]).sum()
        lu_taz_df = lu_taz_df.sort_values(by="ZONE")
        # combine categories
        taz_landuse = pd.DataFrame()
        _land_use_aggregation = self.config.land_use_aggregation
        for total_column, sub_categories in _land_use_aggregation.items():
            taz_landuse[total_column] = lu_taz_df[sub_categories].sum(axis=1)
        taz_landuse.reset_index(inplace=True)
        taz_landuse = taz_landuse.rename(columns={"TOTEMP":"total","RETEMPN":"retail"})
        return taz_landuse

    def validate_inputs(self):
        #TODO
        pass

    def origin_logsum(attraction: NumpyArray, impedance: NumpyArray, decay_factor: float)->NumpyArray:
        """Calculates origin logsum, or the relative accessibility of a given origin zone.
        
        Sums size of attractions, weighted by the impedance between zone and attraction.

        Assumes that impedance of zero indicates a zone is not accessible.  
        Assumes that if the zone is not accessible in either direction, it should not be included 
        in the size calculation.

        Args:
            attraction (NumpyArray): 1-D zone numpy array of an attraction variable
            impedance (NumpyArray): 2-D zone by zone impedance array from origin to destination
            decay_factor (float): factor to dampen the impedances by. Should be less than 1.

        Returns:
            NumpyArray: 1-d zone-based numpy array of the logsum size of the attractions available to
                each zone, weighted by how hard it is to get to the attractions. 
        """

        
        size = attraction * np.exp(decay_factor * impedance)

        # Set size of zero-impedance entries in either direction to zero 
        size = np.where(np.transpose(impedance)==0,0,size)
        size = np.where(impedance==0,0,size)

        #Sum exponentiated all columns for a given row and take the log
        logsum = np.log(size.sum(axis=1)+1.0)
        
        return logsum
    
    def _highway_postprocess(self):
        """
        Temporary fix for now until the zone system is updated.
        """
        root_src_dir = os.path.abspath(self.controller.run_dir)
        skim_path = pathlib.Path(root_src_dir) / self.controller.config.highway.output_skim_path
        hp = HighwayPostprocessor(skim_path, skim_path)
        hp.update_skim_values()

    def _transit_postprocess(self):
        """
        Temporary fix for now until transit assignment uses correct matrix names.
        """
        root_src_dir = os.path.abspath(self.controller.run_dir)
        skim_path = pathlib.Path(root_src_dir) / self.controller.config.transit.output_skim_path
        tp = TransitPostprocessor(skim_path, skim_path)
        tp.update_skim_names()

    @property
    def num_internal_zones(self):
        return len(pd.read_csv(
            self.get_abs_path(self.controller.config.scenario.landuse_file), usecols = [self.controller.config.scenario.landuse_index_column]))


    def _generate_accessibility_file(self):
        modes = [
        'auto',
        'transit',
        'walk'
            ]
        
        time_periods = ['peak','offpeak']
        attraction_type = ['total','retail']
        
        _mode_period_type = itertools.product(modes,time_periods,attraction_type)
        self.zones = self.emme_scenario.zone_numbers
        
        # Ensure input arguments use the number of internal zones.

        num_internal_zones = self.num_internal_zones
        
        for _mode,_period,_type in _mode_period_type:
            formulas = self.config.__dict__[f'formula_{_mode}']
            skim = self.skims(mode=_mode,time_period=_period,skim_prop=formulas['prop'])
            self.logsums_df[f'{self.config.mode_names[_mode]}{_period.title()}{_type.title()}'] = HomeAccessibility.origin_logsum(
                self.employment_df[_type].values,
                skim[:num_internal_zones, :num_internal_zones] ,
                self.config.__dict__[f'dispersion_{_mode}'],
            )
         

         
        self.logsums_df.reset_index(drop = True, inplace = True)
        self.logsums_df.index.name = 'taz'
        self.logsums_df.index = self.logsums_df.index + 1
        
        # drop peak vs. off peak columns for non-motorized
        self.logsums_df.drop(['nonMotorizedOffpeakTotal','nonMotorizedOffpeakRetail'], axis = 1, inplace = True)
        self.logsums_df.rename(columns = {'nonMotorizedPeakTotal':'nonMotorizedTotal','nonMotorizedPeakRetail':'nonMotorizedRetail'}, inplace = True)
        self.logsums_df.to_csv(self.get_abs_path(self.config.outfile))  
    
    def run(self):
    
        # self._highway_postprocess()
        # self._transit_postprocess()

        self._generate_accessibility_file()

    @property
    def estimation_docs(self):
        """Original Estimation Calculation script.

        Part One: Variable definitions
            sovdist ,           {sov distance           }
            sovtime ,           {sov in-vehicle time    }
            sovtoll ,           {sov toll cost          }
            hovdist ,           {hov distance           }
            hovtime ,           {hov in-vehicle time    }
            hovtoll ,           {hov toll cost          }
            wltfare ,           {local transit fare cost          }
            wltwalk ,           {local transit total walk time    }
            wltfwait,           {local transit first wait time    }
            wltxfers,           {local transit transfers          }
            wltxfert,           {local transit transfer time      }
            wltlociv,           {local transit local in-vehicle time}
            wptfare ,           {premium transit fare cost          }
            wptwalk ,           {premium transit total walk time    }
            wptfwait,           {premium transit first wait time    }
            wptxfers,           {premium transit transfers          }
            wptxfert,           {premium transit transfer time      }
            wptlociv,           {premium transit local in-vehicle time}
            wptprmiv,           {premium transit premium in-vehicle time}
            dtwfare ,           {drive to transit fare cost          }
            dtwwalk ,           {drive to transit total walk time    }
            dtwfwait,           {drive to transit first wait time    }
            dtwxfers,           {drive to transit transfers          }
            dtwxfert,           {drive to transit transfer time      }
            dtwlociv,           {drive to transit local in-vehicle time}
            dtwprmiv,           {drive to transit premium in-vehicle time}
            dtwautot,           {drive to transit auto in-vehicle time}
            wtdfare ,           {drive from transit fare cost          }
            wtdwalk ,           {drive from transit total walk time    }
            wtdfwait,           {drive from transit first wait time    }
            wtdxfers,           {drive from transit transfers          }
            wtdxfert,           {drive from transit transfer time      }
            wtdlociv,           {drive from transit local in-vehicle time}
            wtdprmiv,           {drive from transit premium in-vehicle time}
            wtdautot:todmat;    {drive from transit auto in-vehicle time};

        Part Two:  Automobile accessibility calculations
        
            1=AM,2=MD,3=PM,4=NT
            {peak car}
            accm[2]:=accm[2]+(retemp[d]+seremp[d])*exp(-0.05*(sovtime[1,o,d]+sovtime[3,d,o])/100.0);
            accm[1]:=accm[1]+ totemp[d]           *exp(-0.05*(sovtime[1,o,d]+sovtime[3,d,o])/100.0);
            {off-peak car}
            accm[3]:=accm[3]+ totemp[d]           *exp(-0.05*(sovtime[2,o,d]+sovtime[2,d,o])/100.0);
            accm[4]:=accm[4]+(retemp[d]+seremp[d])*exp(-0.05*(sovtime[2,o,d]+sovtime[4,d,o])/100.0);

        Part Three:  Transit accessibility calculations

            {peak transit}
            ltt:=2.0*(wltwalk[1,o,d]+wltfwait[1,o,d]+wltxfert[1,o,d])+wltlociv[1,o,d]
                +2.0*(wltwalk[3,d,o]+wltfwait[3,d,o]+wltxfert[3,d,o])+wltlociv[3,d,o];
            ptt:=2.0*(wptwalk[1,o,d]+wptfwait[1,o,d]+wptxfert[1,o,d])+wptlociv[1,o,d]+wptprmiv[1,o,d]
                +2.0*(wptwalk[3,d,o]+wptfwait[3,d,o]+wptxfert[3,d,o])+wptlociv[3,d,o]+wptprmiv[3,d,o];
            ltt2:=   (wltwalk[1,o,d]+wltfwait[1,o,d]+wltxfert[1,o,d])+wltlociv[1,o,d]
                +(wltwalk[3,d,o]+wltfwait[3,d,o]+wltxfert[3,d,o])+wltlociv[3,d,o];
            ptt2:=   (wptwalk[1,o,d]+wptfwait[1,o,d]+wptxfert[1,o,d])+wptlociv[1,o,d]+wptprmiv[1,o,d]
                +(wptwalk[3,d,o]+wptfwait[3,d,o]+wptxfert[3,d,o])+wptlociv[3,d,o]+wptprmiv[3,d,o];
            
            if (ltt>0) and (ltt<ptt) then begin
                accm[5]:=accm[5]+ totemp[d]           *exp(-0.05*ltt2/100.0);
                accm[6]:=accm[6]+(retemp[d]+seremp[d])*exp(-0.05*ltt2/100.0);
            end else
            if (ptt>0) then begin
                accm[5]:=accm[5]+ totemp[d]           *exp(-0.05*ptt2/100.0);
                accm[6]:=accm[6]+(retemp[d]+seremp[d])*exp(-0.05*ptt2/100.0);
            end;
            
            {off-peak transit}
            ltt:=2.0*(wltwalk[2,o,d]+wltfwait[2,o,d]+wltxfert[2,o,d])+wltlociv[2,o,d]
                +2.0*(wltwalk[2,d,o]+wltfwait[2,d,o]+wltxfert[2,d,o])+wltlociv[2,d,o];
            ptt:=2.0*(wptwalk[2,o,d]+wptfwait[2,o,d]+wptxfert[2,o,d])+wptlociv[2,o,d]+wptprmiv[2,o,d]
                +2.0*(wptwalk[2,d,o]+wptfwait[2,d,o]+wptxfert[2,d,o])+wptlociv[2,d,o]+wptprmiv[2,d,o];
            ltt2:=   (wltwalk[2,o,d]+wltfwait[2,o,d]+wltxfert[2,o,d])+wltlociv[2,o,d]
                +(wltwalk[2,d,o]+wltfwait[2,d,o]+wltxfert[2,d,o])+wltlociv[2,d,o];
            ptt2:=   (wptwalk[2,o,d]+wptfwait[2,o,d]+wptxfert[2,o,d])+wptlociv[2,o,d]+wptprmiv[2,o,d]
                +(wptwalk[2,d,o]+wptfwait[2,d,o]+wptxfert[2,d,o])+wptlociv[2,d,o]+wptprmiv[2,d,o];
            if (ltt>0) and (ltt<ptt) then begin
                accm[7]:=accm[7]+ totemp[d]           *exp(-0.05*ltt2/100.0);
                accm[8]:=accm[8]+(retemp[d]+seremp[d])*exp(-0.05*ltt2/100.0);
            end else
            if (ptt>0) then begin
                accm[7]:=accm[7]+ totemp[d]           *exp(-0.05*ptt2/100.0);
                accm[8]:=accm[8]+(retemp[d]+seremp[d])*exp(-0.05*ptt2/100.0);
            end;

        Part Four:  Non-motorized calculations

            if (sovdist[2,o,d]<=300) then begin
                accm[ 9]:=accm[ 9]+ totemp[d]           *exp(-1.00*(sovdist[2,o,d]+sovdist[2,d,o])/100.0);
                accm[10]:=accm[10]+(retemp[d]+seremp[d])*exp(-1.00*(sovdist[2,o,d]+sovdist[2,d,o])/100.0);
            end;
        """
        pass