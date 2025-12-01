# Velocidad en procesado de datos
Mediapipe ofrece 2 metodos para lograr lo que quiero. Podia usar:
1. El modelo dedicado a procesado de manos y el dedicado al procesado de pose. La ventaja era que no computaba landmarks innecesarios faciales, la desventaja es que tenia que usar 2 modelos
2. Un modelo holistic que procesaba todo. La desventaja era que procesaba muchos landmarks de la cara que aun no tenia planeado usar

## Resultados
Usar los 2 modelos de forma normal tomaban el doble de tiempo que usar holistic. Usar pose + hands corridos en diferentes threads tenia tiempos similares a holistic pero aun asi era un poco mas lento.