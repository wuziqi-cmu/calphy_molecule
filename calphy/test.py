import numpy as np
from matplotlib import pyplot as plt
from ovito.io import import_file, export_file
from ovito.modifiers import ConstructSurfaceModifier, CoordinationAnalysisModifier, TimeAveragingModifier

avgOver = 100
Nbins = 200


# Load a simulation trajectory consisting of several frames:
pipeline = import_file("40000-water-npt.dump")
print("Number of MD frames:", pipeline.source.num_frames)

# Insert the RDF calculation modifier into the pipeline:
pipeline.modifiers.append(CoordinationAnalysisModifier(cutoff = 5.0, number_of_bins = Nbins, partial = True))

# Insert the time-averaging modifier into the pipeline, which accumulates
# the instantaneous DataTable produced by the previous modifier and computes a mean histogram.
# pipeline.modifiers.append(TimeAveragingModifier(operate_on='table:coordination-rdf'))
averageRDFData = np.zeros((avgOver, Nbins, 4))
# Data export method 1: Convert to NumPy array and write data to a text file:
for i in range(avgOver):
    total_rdf = pipeline.compute(i).tables['coordination-rdf'].xy()
    averageRDFData[i] = total_rdf

averageRDFData = np.mean(averageRDFData, axis=0)

plt.plot(averageRDFData[:, 1:])
plt.show()

# np.savetxt("rdf.txt", total_rdf)

# Data export method 2: Use OVITO's own export function for DataTable objects:
# export_file(pipeline, "ov-rdf.txt", "txt/table", key="coordination-rdf")

# for i in zip(a,b):
#     print(i[0], i[1])
#     print(sqrt(2))
pass