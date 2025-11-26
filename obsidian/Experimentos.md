# Experimento 1
## Descripcion
- Una RNN simple de 5 capas ocultas, los videos los elegi procesar a 24fps. 
- Para el processado en tiempo real, lo procesaba en ventanas de 2 segundos.
- Usaba un umbral para el softmax para distinguir si hacia la seña o no estaba haciendo nada
### Resultados
Aunque distinguia las señas correctamente cuando se las hacia, y lograba tener muy buen validation accuracy al entrenarla(~97%), el problema era que el  metodo del umbral no servia para distinguir cuando no se estaban haciendo las señas. Eso es porque para 2 señas muy parecidas(rojo y amarillo por ejemplo), aunque el resultado fuera rojo, tenia accuracy bajo porque el amarillo tambien tenia mucho peso.

Lo que voy a hacer para resolver esto es:
- Bajar los FPS a ~6. Quiero que el procesamiento sea lo mas a corto plazo posible
- Cada seña dura ~2s, asi que si mis señas duran mas de 10 frames, voy a usar una LSTM o una GRU
- Voy a no solo entrenar el modelo con las señas de LSA64, tambien lo voy a entrenar con videos de oradores de charlas TED haciendo cosas con las manos pero no haciendo señas asi distingue entre sus opciones la opcion de no-seña

Otro problema que creo que pueda surgir es que como los de LSA64 estan todos sentados de la misma manera, siento que puede generalizar mal, para eso mi idea(futura) es:
- Tener una red dedicada a predecir si la seña es con una mano, o con las 2. Quizas probar la imagen con y sin espejar y comparar probabilidades para ver si la mano dominante es la zurda, o quizas es algo configurable
- Para la clasificacion de señas:
  - Una red dedicada a clasificar señas de 1 o 2 manos pero al pasarle una seña de una mano, poner en 0 la segunda mano y pasarle un flag de missing
  - 2 redes. Una dedicada a señas de una mano, la otra dedicada a señas de 2 manos

No se donde meteria el logit de "no seña", si en la primera o la segunda etapa.
# Experimento 2
## Descripcion
### Data processing
- 6fps
- Clips de charlas ted, 2s por clip.
- Para la escala de las manos, uso los puntos mano del pose