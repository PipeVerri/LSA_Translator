# Segmentacion en base a aceleracion en manos(incompleto)
El objetivo de este experimento era graficar la aceleracion de los landmarks de las manos para ver que tan viable era armar un modelo de clasificacion no supervisada para la segmentacion de glosas

La idea era medir 2 aceleraciones y usarlas como thresholds, la de los landmarks de la mano respecto a la muñeca, y la de el cuerpo respecto al brazo. Aun asi, antes de completar el experimento vi que no iba a funcionar
## Resultados
Aunque a simple vista parezca viable setear un treshold de aceleracion para la captura de ventanas, aun asi no es viable porque normalmente en LSA(y los lenguajes de señas en general) no hay pausas ni espacios entre las señas que me permitan segmentar facilmente usando la aceleracion

La corriente de segmentacion de señas de forma no supervisada fue reemplazada por los transformers(y ahora por modelos semi-gloss free), asi que parece que no es viable este metodo.

Hay nuevos papers donde aunque usen aceleracion, aun asi la usan como parametro para un modelo encargado de segmentar en vez de hacer la segmentacion usando un algoritmo ad-hoc basado en picos de aceleracion(como aca https://www.researchgate.net/publication/24428978_Sign_Language_Spotting_with_a_Threshold_Model_Based_on_Conditional_Random_Fields)
