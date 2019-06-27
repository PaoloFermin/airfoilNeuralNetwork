from os import path
from os import getcwd 
from PyFoam.RunDictionary.SolutionDirectory import SolutionDirectory
from PyFoam.RunDictionary.ParsedParameterFile import ParsedParameterFile
from PyFoam.Basics.DataStructures import Vector
from PyFoam.Execution.BasicRunner import BasicRunner
from PyFoam.Error import error

from numpy import linspace

#set U values at equal intervals between range
paramValues = linspace(0.1, 0.8, 8)

caseName = "airFoil2D"
solver = "simpleFoam" 

#convert current case directory into a pyfoam-readable object
#create archive directory to store results
dire = SolutionDirectory(caseName, archive=None)

#create log file
logFile = dire.makeFile("ParameterVariationResults")

for param in paramValues:
	#clone template 
	#copies directories 0, constant, and system
	caseName = "airfoil-u%.1f"%param
	case = dire.cloneCase(caseName)
	
	#read correct parameter file and change parameter
	velBC = ParsedParameterFile(path.join(dire.name,"0","U"))
	velBC["internalField"].setUniform(Vector(param,0,0))
	velBC.writeFile()
	
	print("cwd: " + getcwd() )
	#print("case name: " + caseName)	
	
	#run blockmesh
	foamRun = BasicRunner(argv=["simpleFoam", "-case", case.name], logname="simpleFoam.log")
	print("Running simpleFoam")
	foamRun.start()
	if not foamRun.runOK():
		error("There was a problem with simpleFoam")
		

	#run the case
	#argv assumes OpenFOAM convention: <foam command> <dir> <case>
	#run = BasicRunner(argv=[solver,getcwd(),caseName], logname="airFoilLog")	
	#run.start()

	#archive results (from last timestep)
#	dire.lastToArchive("vel=%g"%v)
	logFile.writeLine("test for param=%d"%param)

