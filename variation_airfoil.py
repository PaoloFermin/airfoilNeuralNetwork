from datetime import datetime

start_time = datetime.now()

from os import path, getcwd
from PyFoam.RunDictionary.SolutionDirectory import SolutionDirectory
from PyFoam.RunDictionary.ParsedParameterFile import ParsedParameterFile
from PyFoam.Basics.DataStructures import Vector
from PyFoam.Error import error

#import necessary runners here
from PyFoam.Execution.BasicRunner import BasicRunner

from PyFoam.Applications.Decomposer import Decomposer
from PyFoam.Applications.CaseReport import CaseReport
from PyFoam.Execution.ParallelExecution import LAMMachine
num_procs = 4

from math import cos
from math import sin
from numpy import arange
from numpy import radians

import csv 

#set U values at equal intervals between range
speedOfSound = 340
machs = arange(0.1, 1, 0.1)	#start, stop(exclusive), interval
angles = arange(-10, 11, 1) 	#start, stop(exclusive), interval

templateName = "airFoil2D"
solver = "simpleFoam" 

#convert current case directory into a pyfoam-readable object
#create archive directory to store results
dire = SolutionDirectory(templateName, archive=None)

#create empty log file
log = open("AirfoilParameterVariationLog", "w")
log.close()
logTable = open("results.csv", "w")
writer = csv.writer(logTable)
writer.writerow(['Ux', 'Uy', 'U', 'angle', 'Cd', 'Cl'])
logTable.close()

copy_dir = 'openfoamruns'
base_case = 'airFoil2D'

for mach in machs:
	for angle in angles:
		#clone template 
		#copies directories 0, constant, and system
		clone_name = "/%s/airfoil-u%.1f-a%0.1f" % (path.join(getcwd(), copy_dir), mach, angle)
		clone = dire.cloneCase(clone_name)
		
		Ux = mach * speedOfSound * cos(radians(angle))
		Uy = mach * speedOfSound * sin(radians(angle))

		#read correct parameter file and change parameter
		velBC = ParsedParameterFile(path.join(clone_name,"0","U"))
		velBC["internalField"].setUniform(Vector(Ux, Uy, 0))
		velBC.writeFile()
		
		#edit controlDict to account for change in U
		controlDict = ParsedParameterFile(path.join(clone_name,"system","controlDict"))
		controlDict["functions"]["forcesCoeffs"]["liftDir"] = Vector(-sin(radians(angle)), cos(radians(angle)), 0)
		controlDict["functions"]["forcesCoeffs"]["dragDir"] = Vector(cos(radians(angle)), sin(radians(angle)), 0)
		controlDict["functions"]["forcesCoeffs"]["magUInf"] = mach * speedOfSound
		controlDict.writeFile()	
	
		#implement parallelization
		print('Decomposing...')
		Decomposer(args=['--progress', clone_name, num_procs])
		CaseReport(args=['--decomposition', clone_name])
		machine = LAMMachine(nr=num_procs)

		#run simpleFoam
		foamRun = BasicRunner(argv=[solver, "-case", clone_name], logname="simpleFoam")
		print("Running simpleFoam")
		foamRun.start()
		if not foamRun.runOK():
			error("There was a problem with simpleFoam")
		
		#get headers and last line of postprocessing file
		with open(path.join(clone_name, 'postProcessing','forcesCoeffs','0','coefficient.dat'), "rb") as table:
			last = table.readlines()[-1].decode()
			print("last line of coefficients" + last)		
			splitLast = last.split()
			print(splitLast)	
			Cd = float(splitLast[2])
			Cl = float(splitLast[3])
			table.close()

		with open("results.csv", "a") as logTable: 
			output = [Ux, Uy, mach*speedOfSound, angle, Cd, Cl]
			writer = csv.writer(logTable)
			writer.writerow(output)
			logTable.close()

print('Execution time: {}'.format(datetime.now() - start_time))
