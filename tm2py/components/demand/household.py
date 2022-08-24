"""Placeholder docstring for CT-RAMP related components for household residents' model."""

import shutil as _shutil

from tm2py.components.component import Component
from tm2py.logger import LogStartEnd
from tm2py.tools import run_process


class HouseholdModel(Component):
    """Run household resident model."""

    def validate_inputs(self):
        """Validates inputs for component."""
        pass

    @LogStartEnd()
    def run(self):
        """Run the the household resident travel demand model.

        Steps:
            1. Starts household manager.
            2. Starts matrix manager.
            3. Starts resident travel model (CTRAMP).
            4. Cleans up CTRAMP java.
        """
        self._start_household_manager()
        self._start_matrix_manager()
        self._start_jppf_driver()
        self._start_jppf_node0()
        self._run_resident_model()
        self._stop_java()

    def _start_household_manager(self):
        commands = [
            "cd /d {}".format(self.controller.run_dir.__str__()),
            "CALL {}\\CTRAMP\\runtime\\SetPath.bat".format(self.controller.run_dir.__str__()),
            "set HOST_IP={}".format(self.controller.config.run.host_ip_address),
            "start \"Household Manager\" java -Xms20000m -Xmx20000m -Dlog4j.configuration=log4j_hh.xml com.pb.mtc.ctramp.MtcHouseholdDataManager -hostname %HOST_IP%",
            "echo Hello World"
        ]
        run_process(commands, name="start_household_manager")

    def _start_matrix_manager(self):
        commands = [
            "cd /d {}".format(self.controller.run_dir.__str__()),
            "CALL {}\\CTRAMP\\runtime\\SetPath.bat".format(self.controller.run_dir.__str__()),
            "set HOST_IP={}".format(self.controller.config.run.host_ip_address),
            "start \"Matrix Manager\" java -Xms14000m -Xmx140000m -Dlog4j.configuration=log4j_mtx.xml -Djava.library.path=\"CTRAMP/runtime\" com.pb.models.ctramp.MatrixDataServer -hostname %HOST_IP%",
        ]
        run_process(commands, name="start_matrix_manager")

    def _run_resident_model(self):
        sample_rate_iteration = {1: 0.3, 2: 0.5, 3: 1, 4: 0.02, 5: 0.02}
        iteration = self.controller.iteration
        sample_rate = sample_rate_iteration[iteration]
        seed = 0
        #_shutil.copyfile("CTRAMP\\runtime\\mtctm2.properties", "mtctm2.properties")
        commands = [
            "cd /d {}".format(self.controller.run_dir.__str__()),
            "CALL {}\\CTRAMP\\runtime\\SetPath.bat".format(self.controller.run_dir.__str__()),
            "java -showversion -Xmx6000m -cp %CLASSPATH% -Dlog4j.configuration=log4j.xml -Djava.library.path=%RUNTIME% -Djppf.config=jppf-clientDistributed.properties "
            "com.pb.mtc.ctramp.MtcTourBasedModel mtcTourBased -iteration {} -sampleRate {} -sampleSeed {}".format(iteration, sample_rate, seed),
        ]
        run_process(commands, name="run_resident_model")
    
    def _start_jppf_driver(self):
        commands = [
            "cd /d {}".format(self.controller.run_dir.__str__()),
            "CALL {}\\CTRAMP\\runtime\\SetPath.bat".format(self.controller.run_dir.__str__()),
            "set HOST_IP={}".format(self.controller.config.run.host_ip_address),
            "start \"JPPF Server\" java -server -Xmx16m -Dlog4j.configuration=log4j-driver.properties -Djppf.config=jppf-driver.properties org.jppf.server.DriverLauncher",
        ]
        run_process(commands, name="start_jppf_driver")
    
    def _start_jppf_node0(self):
        commands = [
            "cd /d {}".format(self.controller.run_dir.__str__()),
            "CALL {}\\CTRAMP\\runtime\\SetPath.bat".format(self.controller.run_dir.__str__()),
            "set HOST_IP={}".format(self.controller.config.run.host_ip_address),
            "start \"Node 0\" java -server -Xmx128m -Dlog4j.configuration=log4j-node0.xml -Djppf.config=jppf-node0.properties org.jppf.node.NodeLauncher",
        ]
        run_process(commands, name="start_jppf_node")

    def _stop_java(self):
        run_process(['taskkill /im "java.exe" /F'])
