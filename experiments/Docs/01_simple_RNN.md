# Experimento 1: RNN simple(SimpleDetector)

## Dataset y setup

Quería empezar con algo manejable, así que usé **LSA64**:
- 64 señas distintas.
- ~50 samples por seña.

La idea era:

- Entrenar una **RNN** que recibe como input cada frame de *landmarks* (pose/manos),
- El output es un **softmax sobre las 64 señas posibles**.
- Para el caso continuo:
  - Le voy pasando frames a medida que se generan.
  - Solo considero que “se hizo una seña” cuando alguna clase pasa cierto **threshold** de probabilidad.

Además, probé una versión más explícita en tiempo real:

- Procesaba a **6 fps**.
- Ventanas de **2 segundos**.
- Umbral sobre softmax para distinguir “seña” vs “no seña”.

### ¿Por qué una RNN en vez de GRU/LSTM?

Quería probar lo más simple posible.  
Las dependencias temporales eran cortas (~12 steps), dentro de lo manejable para una RNN vanilla.

## Arquitectura y parámetros

- RNN con **5 capas ocultas**.
- Ancho de **144** (landmarks por frame).
- **100 epochs**, LR \(10^{-4}\), L2 \(10^{-3}\).

## Prueba 1 — Señales aisladas

- Train accuracy > **99%**  
- Validation accuracy ~ **97%**

Muy buen resultado para un dataset chico.  
**Pero fallaba en continuo**:

- Nunca vio secuencias largas,
- No sabía olvidar,
- No sabía identificar “no-seña”.

## Prueba 2 — Ventanas deslizantes

Mejoró un poco, pero seguía sin segmentar bien:

1. **Segmentación rota** → si la ventana cortaba una seña, fallaba.  
2. **Softmax siempre elige alguna seña** → no existe noción de “nothing”.

### Intento: clase extra “no seña”

Agregué:

- clips de charlas TED,
- clase 65 = “no-seña”.

Pero:

- El modelo aprendió “no-seña = manos como orador TED”.
- No generalizó a transiciones reales.

## Conclusiones del experimento

- Sin segmentación temporal real, no funciona.  
- RNN simple no maneja bien transiciones.
- Las ventanas fijas no sirven: la duración de las señas varía demasiado.

## Siguientes pasos derivados

- Bajar FPS a ~6.
- Usar GRU/LSTM.
- Incluir clips variados de “no-seña”.
- Lidiar con:
  - diferencia entre señas de 1 o 2 manos,
  - normalización de posiciones,
  - generalización a entornos reales.

# Experimento 2: RNN más expresiva y más datos

La idea fue mejorar el enfoque del experimento 1 manteniendo la estructura.

## Data processing

- Procesado a **6 fps**.
- Clips de charlas TED de **2s** para “no-seña”.
- Normalización basada en landmarks de mano.

## Modelo

- Igual que antes pero más ancho:
  - **300 unidades por capa**.
  - Más profundidad → overfit sin mejora,
  - Más ancho → sí ayudó.

## Resultados

- El modelo clasifica “no-seña” **solo si las manos parecen de charla TED**.  
  Está fuertemente sesgado.
- Las señas repetidas las toma bien,
- Pero los frames intermedios entre seña y no-seña NO encajan en ninguna clase → clasifica “alguna seña”.

### Conclusión

> El modelo debe aprender **segmentación temporal**, no solo clasificación por ventanas.  
> Si no, nunca va a funcionar en continuo.