# Internship : Évolution de la neige en période de fonte : modélisation et observation au Col du Lautaret
**Abstract**
<p align="justify">
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
<p>
    
**Scientific context:**
<p align="justify">
Snow is a key component of the global climate system. It continuously covers glaciers and ice sheets, and seasonally covers nearly one-third of the Earth’s land surface (Sturm et Liston, 2021). Snow regulates surface energy exchanges with its high albedo and low thermal conductivity. It acts as the primary source of mass for ice sheets and glacier, influencing sea level and representing a freshwater resource downstream of many mountain regions (Viviroli et al., 2007). However, observations show a gradual decline in both the extent and duration of seasonal snow cover, indicating earlier snow melt onset (Bormann et al., 2018). Meltwater is also increasingly observed on ice sheets where it contributes to mass loss either directly through runoff or indirectly via interactions with ice dynamics (Amory et al., 2024). In this context, liquid water is emerging as a central driver of
snowpack evolution and surface energy and water budgets at the planetary scale.
Understanding the behavior of liquid water within snow requires a detailed view of
snow physical properties and their evolution. 
<p>
<p align="justify">
Snow is a dynamic porous medium composed of ice grains and air, whose microstructure (defined as the size, shape, and arrangement of grains and bonds) evolves with time under the action of humidity, temperature and pressure gradients, a process known as metamorphism. The percolation of liquid water depends on this microstructure and, in turn, water alters it by refreezing and warming the snowpack. Meltwater also influences albedo through metamorphism, affecting the energy balance of the snow and, consequently, the rate of melt. These interdependent processes give the snowpack a "memory," where present conditions are shaped by past events, complicating its representation in models. For example, capillary barriers (i.e., interfaces between tow porous layers of contrasting pore sizes, visible as horizontal lay- ers filled with colored water in the picture on the front page of this report) can delay or block percolation, causing water to accumulate above the interface. Refreezing at these locations can form ice crusts, reducing local porosity and affecting water retention capacity. Such complexities underscore the challenges involved in modelling snow-water interactions (Verjans et al., 2019).
<p>
<p align="justify"> 
Snow models incorporate water percolation processes using approaches of varying
complexity. The Crocus model, developed at the Centre d’Études de la Neige (CEN),
simulates snowpack evolution in detail and recently integrated the Richards formalism
(Richards, 1931) as a physically-based alternative to the widely used "bucket" scheme
(D’Amboise et al., 2017). In the bucket approach, snow layers are treated as reservoirs
with fixed water retention capacity and excess water is passed downward. While the
Richards-based approach provides a more physically detailed representation of water
percolation based on the snowpack’s physical properties, it remains computationally
intensive and has yet to be fully evaluated. Regardless of their complexity, all these modeling approaches require detailed field observations to assess their ability to accurately
represent water content, pathways, and infiltration depth (Wever et al., 2015).
<p>

**Objectives and research questions**
<p align="justify">
This internship aims to evaluate different representations of water percolation in snow
using field observations from an alpine site. The work is structured around two main
objectives: (1) building a dataset for evaluating percolation schemes, and (2) applying this dataset to the Crocus model. By comparing simulations with field data, the study
seeks to identify key processes and assess the benefits of using more physically detailed
approaches. Ultimately, this work aims to address the following research questions:

i) How to measure liquid water content into the snow? The challenge lies in achieving
measurements that are qualitative (detecting the presence or absence of liquid water),
quantitative (estimating the liquid water content), continuous (autonomous over time),
and vertically resolved (to capture water content gradients and infiltration depth).

ii) What are the differences and performance characteristics of the Bucket and
Richards percolation schemes for water flow in snow?
<p>
    
## Contributors 
These codes were created by Glenn Pitiot under the direction of Charles Amory and Kévin Fourteau.
## Material
<p align="justify">
This repository is organized into the following categories:

- **Report:** Contains the final report delivered at the end of the internship.  
- **Notebooks:** Includes all materials related to methods and results.  
- **Scripts and Utils:** Contains all the code required to run the notebooks.
- **Data:** All data used in this work were first observed during the 2024–2025 snow season at Col du Lautaret. Meteorological data were collected by FluxAlps (https://www.osug.fr/le-labex/actions-soutenues/recherche/Ecologie/fluxalp-equipement-d-un-site-alpin-pour-la-mesure-des-flux-d-energie-et-de.html). Finally, outputs from Crocus simulations were also used.
<p>

