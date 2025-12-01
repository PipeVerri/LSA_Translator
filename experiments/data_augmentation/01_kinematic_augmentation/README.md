# Data augmentation a traves de ruido y control kinematico(incompleto)
El mayor desafio en SLT es el de la falta de datos y la gran variabilidad que hay entre señantes. Para eso una de mis ideas era
1. De alguna forma separar la seña en movimientos basicos como "mano derecha hacia arriba-izquierda, mano izquierda hacia abajo-derecha", tambien podria hacer una separacion de movimientos basicos por dedos aparte de la palma
2. Representar ese movimiento con un vector
3. Para cada vector $\mathbf{v}$, generar una direccion de ruido $\mathbf{v}\circ \mathbf{x}$ siendo el vector $\mathbf{x}\sim\mathcal{N}(0, \sigma^2 I)$
4. Sumarle a $\mathbf{v}$ su vector de ruido
5. Mover el resto de los landmarks definiendo una cadena kinematica de huesos, similar a como se hace en animacion.

La idea detras de esto es que si en un movimiento basico me muevo 10cm para arriba y 1cm para la derecha, entonces tendria sentido que luego de agregarle ruido me quede algo como 12cm para arriba y 2cm para la derecha, no 11cm para arriba y 7cm para la derecha.

