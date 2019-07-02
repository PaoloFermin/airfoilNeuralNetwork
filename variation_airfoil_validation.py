from os import path
from os import getcwd 
from PyFoam.RunDictionary.SolutionDirectory import SolutionDirectory
from PyFoam.RunDictionary.ParsedParameterFile import ParsedParameterFile
from PyFoam.Basics.DataStructures import Vector
from PyFoam.Error import error

#import necessary runners here
from PyFoam.Execution.BasicRunner import BasicRunner
from PyFoam.Execution.ConvergenceRunner import ConvergenceRunner

#import necessary analyzers here
from PyFoam.LogAnalysis.BoundingLogAnalyzer import BoundingLogAnalyzer

from math import cos
from math import sin
from numpy import arange
from numpy import radians

import csv 

#set U values at equal intervals between range
speedOfSound = 340
machs = arange(0.1, 1, 0.1)	#start, stop(exclusive), interval
angles = [-20, -15, 15, 20] 	#start, stop(exclusive), interval

templateName = "airFoil2D"
solver = "simpleFoam" 

#convert current case directory into a pyfoam-readable object
#create archive directory to store results
dire = SolutionDirectory(templateName, archive=None)

#create empty log file
log = open("AirfoilParameterVariationLog", "w")
log.close()
logTable = open("validation_results.csv", "w")
writer = csv.writer(logTable)
writer.writerow(['Ux', 'Uy', 'U', 'angle', 'Cd', 'Cl'])
logTable.close()

for mach in machs:
	for angle in angles:
		#clone template 
		#copies directories 0, constant, and system
		cloneName = "airfoil-u%.1f-a%0.1f"% (mach, angle)
		clone = dire.cloneCase(cloneName)
		
		Ux = mach * speedOfSound * cos(radians(angle))
		Uy = mach * speedOfSound * sin(radians(angle))

		#read correct parameter file and change parameter
		velBC = ParsedParameterFile(path.join(clone.name,"0","U"))
		velBC["internalField"].setUniform(Vector(Ux, Uy, 0))
		velBC.writeFile()
		
		#edit controlDict to account for change in U
		controlDict = ParsedParameterFile(path.join(clone.name,"system","controlDict"))
		controlDict["functions"]["forcesCoeffs"]["liftDir"] = Vector(-sin(radians(angle)), cos(radians(angle)), 0)
		controlDict["functions"]["forcesCoeffs"]["dragDir"] = Vector(cos(radians(angle)), sin(radians(angle)), 0)
		controlDict["functions"]["forcesCoeffs"]["magUInf"] = mach * speedOfSound
		controlDict.writeFile()	
	
		print("cwd: " + getcwd() )
		#print("case name: " + caseName)	
		
		#run blockmesh
		foamRun = BasicRunner(argv=[solver, "-case", clone.name], logname="simpleFoam.log")
		print("Running simpleFoam")
		foamRun.start()
		if not foamRun.runOK():
			error("There was a problem with simpleFoam")
		
		#get headers and last line of postprocessing file
		with open("./" + cloneName + "/postProcessing/forcesCoeffs/0/coefficient.dat", "rb") as table:
			last = table.readlines()[-1].decode()
			print("last line of coefficients" + last)		
			splitLast = last.split()
			print(splitLast)	
			Cd = float(splitLast[2])
			Cl = float(splitLast[3])
			table.close()

		with open("AirfoilParameterVariationLog", "a") as log:
			log.write('\nUx = %0.4f, Uy = %0.4f, Cd = %0.4f, Cl = %0.4f' % (Ux, Uy, Cd, Cl))
			log.close()

		with open("validation_results.csv", "a") as logTable: 
			output = [Ux, Uy, mach*speedOfSound, angle, Cd, Cl]
			writer = csv.writer(logTable)
			writer.writerow(output)
			logTable.close()

