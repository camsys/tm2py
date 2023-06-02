"""Placeholder docstring for CT-RAMP related components for household residents' model."""

import shutil as _shutil
import os, pathlib, itertools

from tm2py.components.component import Component
from tm2py.logger import LogStartEnd
from tm2py.tools import run_process
from tm2py.components.demand.prepare_demand import PrepareHighwayDemand


class HouseholdModel(Component):
    """Run household resident model."""

    def validate_inputs(self):
        """Validates inputs for component."""
        pass

    @LogStartEnd()
    def run(self):
        """Run the the household resident travel demand model.

        Steps:
            1. Moves inputs to CT-RAMP directory if ctramp_run_dir is different from the main model directory
            2. Updates telecommute constants
            3. Starts household manager.
            4. Starts matrix manager.
            5. Starts resident travel model (CTRAMP).
            6. Cleans up CTRAMP java.
            7. Produce household demand matrices for highway and transit assignment.
        """
        if self.controller.config.household.copy_from_examples:
            self._move_inputs_to_run_dir()
        self._update_telecommute_constants()
        self._start_household_manager()
        self._start_matrix_manager()
        self._start_jppf_driver()
        self._start_jppf_node0()
        self._run_resident_model()
        self._stop_java()
        self._prepare_demand_for_assignment()
        
    def _move_inputs_to_run_dir(self):
        if not os.path.samefile(self.controller.config.household.ctramp_run_dir,
            os.path.abspath(self.controller.run_dir)):
            
            root_src_dir = os.path.abspath(self.controller.run_dir)
            root_dst_dir = self.controller.config.household.ctramp_run_dir

            # Folders that should already exist: CT-RAMP, popsyn, logsums, INPUT\\params.properties
            # Folders that need to be populated: landuse once; taz centroid file for skim postprocessing
            # Folders taht need to be updated every iteration: skims (including accessibility.csv)
            
            # Move land use file
            dst_dir = pathlib.Path(root_dst_dir) / 'landuse'
            if not os.path.exists(dst_dir):
                os.mkdir(dst_dir)
            
            src_file = self.get_abs_path(
                pathlib.Path(root_src_dir) / self.controller.config.scenario.landuse_file
            )
            
            dst_file = pathlib.Path(dst_dir) / os.path.basename(src_file)
            if not os.path.exists(dst_file):
                _shutil.copy(src_file, dst_file)
            
            # Move skims
            dst_dir = pathlib.Path(root_dst_dir) / 'skims'
            if not os.path.exists(dst_dir):
                os.mkdir(dst_dir)
            
            # Nonmotorized skim
            src_file = (pathlib.Path(root_src_dir) / self.controller.config.active_modes.output_skim_path / 
                self.controller.config.active_modes.output_skim_filename_tmpl)
            
            dst_file = pathlib.Path(dst_dir) / os.path.basename(src_file)
            if os.path.exists(dst_file):
                os.remove(dst_file)
            _shutil.copy(src_file, dst_file)
                
            
            # Transit skims
            periods = [c.name for c in self.controller.config.time_periods]
            trn_classes = [c.name for c in self.controller.config.transit.classes]
            for _period, _class in itertools.product(periods, trn_classes):
                src_file = (pathlib.Path(root_src_dir) / self.controller.config.transit.output_skim_path / 
                    self.controller.config.transit.output_skim_filename_tmpl.format(
                        time_period = _period, set_name = _class)
                        )
                dst_file = pathlib.Path(dst_dir) / os.path.basename(src_file)
                if os.path.exists(dst_file):
                    os.remove(dst_file)
                _shutil.copy(src_file, dst_file)
            
            # Highway skims
            periods = [c.name for c in self.controller.config.time_periods]
            for _period  in periods:
                src_file = (pathlib.Path(root_src_dir) / self.controller.config.highway.output_skim_path / 
                    self.controller.config.highway.output_skim_filename_tmpl.format(
                        time_period = _period)
                        )
                dst_file = pathlib.Path(dst_dir) / os.path.basename(src_file)
                if os.path.exists(dst_file):
                    os.remove(dst_file)
                _shutil.copy(src_file, dst_file)
            
            # Accessibility file
            src_file = pathlib.Path(root_src_dir) / self.controller.config.accessibility.outfile
            dst_file = pathlib.Path(dst_dir) / os.path.basename(src_file)
            if os.path.exists(dst_file):
                os.remove(dst_file)
            _shutil.copy(src_file, dst_file)
            
            # taz_centroids.csv
            src_file = pathlib.Path(root_src_dir) / 'inputs/hwy/taz_centroids.csv'
            dst_file = dst_dir = pathlib.Path(root_dst_dir) / 'skims/taz_centroids.csv'
            if not os.path.exists(dst_file):
                _shutil.copy(src_file, dst_file)

            
    def _prepare_demand_for_assignment(self):
        prep_demand = PrepareHighwayDemand(self.controller)
        prep_demand.prepare_household_demand()
        

    def _start_household_manager(self):
        commands = [
            "cd /d {}".format(self.controller.config.household.ctramp_run_dir),
            "CALL {}\\CTRAMP\\runtime\\SetPath.bat".format(self.controller.config.household.ctramp_run_dir),
            "set HOST_IP={}".format(self.controller.config.household.host_ip_address),
            "start \"Household Manager\" java -Xms20000m -Xmx20000m -Dlog4j.configuration=log4j_hh.xml com.pb.mtc.ctramp.MtcHouseholdDataManager -hostname %HOST_IP%",
            "echo Hello World"
        ]
        run_process(commands, name="start_household_manager")

    def _start_matrix_manager(self):
        commands = [
            "cd /d {}".format(self.controller.config.household.ctramp_run_dir),
            "CALL {}\\CTRAMP\\runtime\\SetPath.bat".format(self.controller.config.household.ctramp_run_dir),
            "set HOST_IP={}".format(self.controller.config.household.host_ip_address),
            "start \"Matrix Manager\" java -Xms14000m -Xmx140000m -Dlog4j.configuration=log4j_mtx.xml -Djava.library.path=\"CTRAMP/runtime\" com.pb.models.ctramp.MatrixDataServer -hostname %HOST_IP%",
        ]
        run_process(commands, name="start_matrix_manager")

    def _run_resident_model(self):
        sample_rate_iteration_values = self.controller.config.household.sample_rate_iteration
        if self.controller.config.run.start_iteration != 0:
            if len(sample_rate_iteration_values) != self.controller.config.run.end_iteration - self.controller.config.run.start_iteration + 1:
                raise Exception("The length of sample rate values does not match the number of iterations.")
            sample_rate_iteration = dict(zip(
                range(self.controller.config.run.start_iteration, self.controller.config.run.end_iteration + 1), 
                sample_rate_iteration_values))
        else:
            if len(sample_rate_iteration_values) != self.controller.config.run.end_iteration - self.controller.config.run.start_iteration:
                raise Exception("The length of sample rate values does not match the number of iterations.")
            sample_rate_iteration = dict(zip(
                range(self.controller.config.run.start_iteration + 1, self.controller.config.run.end_iteration + 1), 
                sample_rate_iteration_values))
        
        iteration = self.controller.iteration
        sample_rate = sample_rate_iteration[iteration]
        seed = 0
        
        commands = [
            "cd /d {}".format(self.controller.config.household.ctramp_run_dir),
            "CALL {}\\CTRAMP\\runtime\\SetPath.bat".format(self.controller.config.household.ctramp_run_dir),
            "java -showversion -Xmx6000m -cp %CLASSPATH% -Dlog4j.configuration=log4j.xml -Djava.library.path=%RUNTIME% -Djppf.config=jppf-clientDistributed.properties "
            "com.pb.mtc.ctramp.MtcTourBasedModel mtcTourBased -iteration {} -sampleRate {} -sampleSeed {}".format(iteration, sample_rate, seed),
        ]
        run_process(commands, name="run_resident_model")
    
    def _start_jppf_driver(self):
        commands = [
            "cd /d {}".format(self.controller.config.household.ctramp_run_dir),
            "CALL {}\\CTRAMP\\runtime\\SetPath.bat".format(self.controller.config.household.ctramp_run_dir),
            "set HOST_IP={}".format(self.controller.config.household.host_ip_address),
            "start \"JPPF Server\" java -server -Xmx16m -Dlog4j.configuration=log4j-driver.properties -Djppf.config=jppf-driver.properties org.jppf.server.DriverLauncher",
        ]
        run_process(commands, name="start_jppf_driver")
    
    def _start_jppf_node0(self):
        commands = [
            "cd /d {}".format(self.controller.config.household.ctramp_run_dir),
            "CALL {}\\CTRAMP\\runtime\\SetPath.bat".format(self.controller.config.household.ctramp_run_dir),
            "set HOST_IP={}".format(self.controller.config.household.host_ip_address),
            "start \"Node 0\" java -server -Xmx128m -Dlog4j.configuration=log4j-node0.xml -Djppf.config=jppf-node0.properties org.jppf.node.NodeLauncher",
        ]
        run_process(commands, name="start_jppf_node")

    def _stop_java(self):
        run_process(['taskkill /im "java.exe" /F'])
    
    #def _update_telecommute_constants(self):
    #    telecommute_parameters = {'ITER':0, 'MODEL_YEAR': 2015, 'MODEL_DIR': self.controller.config.household.ctramp_run_dir}
    #    commands = [
    #    "cd /d {}".format(self.controller.config.household.ctramp_run_dir),
    #    "CALL {}\\CTRAMP\\runtime\\SetPath.bat".format(self.controller.config.household.ctramp_run_dir),
    #    ] + [
    #    f"set {key}={telecommute_parameters[key]}" for key in telecommute_parameters
    #    ] + [
    #    "%PYTHON_PATH%\\python {}\\CTRAMP\\scripts\\preprocess\\updateTelecommuteConstants.py".format(self.controller.config.household.ctramp_run_dir),
    #    "copy /Y main\\telecommute_constants_00.csv main\\telecommute_constants.csv"
    #    ]
    #    run_process(commands, name="update_telecommute_constants")