# Intership : Évolution de la neige en période de fonte : modélisation et observation au Col du Lautaret

The ongoing development of snow models requires the study of wet and melting
snowpacks. Identifying the key mechanisms at play for water percolation within a natural
snowpack is thus critical to model this complex process. To that end, I participated in
the set-up of an field campaign during the melting season 2024–2025 at Col du Lautaret
in the Alps. Its goal is to provide a clearer picture of water flow in a natural snowpack
and to foster instrumental development. The deployment of continuous temperature and
liquid water content measurements, alongside snow pit profiles carried out during the
melt period, was used to study the percolation dynamics. It was further used to analyze
the performance of two percolation schemes, the bucket scheme and Richards equation,
in the detailed snowpack model Crocus. The Richards percolation scheme theoretically
allows a better modelling of the complexity of percolation compared to the Bucket
scheme, an idealized representation of percolation, but is more computationally costly.
Unfortunately, results show that the commercial liquid water sensors selected for this
internship are inadequate for snow monitoring. The rest of field observations suggest the
important role of capillary barriers and refreezing crusts in influencing the water perco-
lation dynamics. However, the Richards scheme did not provide a better representation
of these barriers compared to the Bucket scheme, raising questions about the benefit of a
more complex scheme. It appears that a key missing mechanism in our modelling attempt
is percolation along preferential paths, leading to a poor representation of the temperature
profiles in Crocus for both schemes. Further development are thus two folds : developing
measurements able to continuously measure liquid water content in snow, and tailoring
numerical schemes to capture the dominant mechanisms at play during percolation.

## Contributors 
These codes were created by Glenn Pitiot under the direction of Charles Amory and Kévin Fourteau.
## Material

This repository is organized into the following categories:

- **Report:** Contains the final report delivered at the end of the internship.  
- **Notebooks:** Includes all materials related to methods and results.  
- **Scripts and Utils:** Contains all the code required to run the notebooks.

## Licence
...
