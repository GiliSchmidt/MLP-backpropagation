----------- Executar -------------
O ponto de entrada para executar o código é o arquivo main.py.
Nele são realizadas a configurações da execução de treinamento e predição.




----------- Treinamento -------------
Para treinar a rede é necessário informar o arquivo com os dados de treinamento e parâmetros.
Isso é feito na função train(net).

Ao treinar, os "weights" de cada camada são salvos em arquivos .txt, na pasta weights.
Além disso, é feito um teste com todos os dados de teste após o treinamento.

O arquivo de treinamento deve estar no seguinte formato, podendo conter N linhas:

[codigo do número][resultado esperado]

Por exemplo:
0 1 0 1 1 0 0 1 0 0 1 0 1 1 1 0 0 1 0

Onde,
representa o desenho do número 1    representa o número 1 em binário
[0 1 0 1 1 0 0 1 0 0 1 0 1 1 1]     [0 0 1 0]



----------- Predição -------------
Para a predição é possível ler os "weights" obtidos em um treinamento anterior.

É possível informar como entrada para a predição 2 formatos de arquivos:

binário, onde o mesmo possui N linhas, cada uma no formato a seguir, representa o desenho do número:
0 1 0 1 1 0 0 1 0 0 1 0 1 1 1

ou o "desenho" do número, onde cada desenho é colocado lado a lado, conforme a seguir:

 X  XX  XXX X X XXX  XX XXX XXX XXX XXX
XX    X   X X X X   X     X X X X X X X
 X   X   XX XXX XXX XX    X XXX XXX X X
 X  X     X   X   X X X   X X X   X X X
XXX XXX XXX   X XXX  X    X XXX   X XXX

Cada número deve ser desenhado em uma matrix de 3x5, podendo desenhar n números, todos lado a lado