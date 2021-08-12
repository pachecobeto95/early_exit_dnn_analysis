# Análise de DNNs com saídas antecipadas
Este código contém todos os códigos desenvolvidos para analisar DNNs com saídas antecipadas considerando uma infraestrutura de computação em borda e nuvem.


## Breve Guia
Este breve guia explica rapidamente o código desenvolvido até o momento com objetivo de auxiliar os demais membros do projeto a prosseguir com os próximos passos

### Comunicação entre Borda e nuvem

Os arquivos edgeWebAPI.py e cloudWebAPI.py, quando executados, permitem iniciar os server da borda e da nuvem respectivamente. Uma vez iniciados, o arquivo end_node_json.py envia 
uma mensagem em formato json apenas para testar. O importante é o arquivo end_node_img.py que permite enviar imagens do daatset Caltech. Para isso, é necessário baixar o dataset,
criar um diretório datasets, e colocar o dataset dentro desse diretório. Após enviar uma imagem, a borda a recebe, faz uma transformação nos dados de entrada para redimensioná-lo,
e o insere na DNN para fazer a inferência. Caso a inferência atinja um critério de confiança p_tar, a amostra é classificada na borda, caso contrário, a borda envia as dados 
intermediários da DNN para a nuvem que termina de executar a inferência. 

Os código desenvolvidos foram adaptados do meu último trabalho do globecom. Portanto, só adaptei e testei para funcionar. Claro que quando treinarmos os nossos modelos para 
realizar as análises desse trabalho teremos que fazer algumas mudanças também. Por exemplo, gente pode adicionar um mecanismo que permita alterar a quantidade de saídas antecipadas
processadas na borda. Em resumo, está funcionando a comunicação entre borda e nuvem como forma de exemplo, depois que treinarmos os nossos modelos para os datasets, gente volta 
e adapta par fazer so experimentos práticos


### Experiments

No diretório experiments, eu deixei um código para carregar os datasets caltech-256, cifar10 e cifar100 e dividí-los em conjunto de treino, valdiação e test conforme uma percentual 
fornecido pelo desenvolvedor. O datasets cifar10 e cifar100 são disponibilizados pelo framework pytorch, enquanto o dataset caltech-256 é necessário baixar separadamente. 
Para baixar acesse os links:
* https://www.kaggle.com/jessicali9530/caltech256
* http://www.vision.caltech.edu/Image_Datasets/Caltech256/

Além disso, fiz um código do arquivo early_exit_dnn_framework.ipynb e early_exit_dnn_framework2.ipynb (google collab) que são duas implementações para uma espécie de framework para DNNs com saídas antecipadas
Na real, esses códigos facilitam a implementação de inserir saídas antecipadas na arquitetura. Vocês poderão notar que está implementadas as arquiteturas AlexNet mobilenet v2 e resnet18
e a inserção das saídas antecipadas ocorrem de acordoc om uma distribuição. O núemros de ramos é dado como parâmetro mas que variaremos ao longo das análises. 

Por fim, também fiz um código de treinamento e validação dos modelos. Caso os membros decisam implementar uma outra métrica de desempenho do modelo, a gente conversa. 

Peço que vão olhando e testando os códigos. Vão surgir dúvidas e tirarei mais rápido possível. Claro que é possível de ter erros de implementação ou algo que gente possa melhorar
faz parte do desenvolvimento corrigir rs. 


# Roteiro

## Treinamento

Para realizar o treinamento, execute o arquivo "train_validation_early_exit_dnn_mbdi". 

Primeiramente, vou descrever as classes implementadas.

LoadDataset -> tem como objetivo carregar o dataset a ser utilizado para treinar, validar e testar o modelo. Essa classe é composta por métodos correpspondente a cada um dos datasets implementado. O método caltech256 carrega e disponibiliza o dataset Caltech 256, enquanto o método cifar_10 e cifar_100 carregam os datasets Cifar 10 e Cifar 100, respectivamente. Estamos com pressa, mas sugiro que leiam rapidamente cada dataset mencionado.    

Os métodos da classe LoadDataset:
* cifar_100(self, root_path, split_ratio):  
    * root_path: esse método faz o download do dataset cifar_100 e armazena no caminho indicado por root_path
    * split_ratio: esse decimal (0 a 1) indica a taxa de divisão do conjunto de dados de treinamento e de validação   
* cifar_10(self, root_path, split_ratio)
    * root_path: esse método faz o download do dataset cifar_100 e armazena no caminho indicado por root_path
    * split_ratio: esse decimal (0 a 1) indica a taxa de divisão do conjunto de dados de treinamento e de validação   
* caltech_256(self, root_path, split_ratio, savePath_idx)
    * root_path: este dataset não é baixado. Você deve baixá-lo pelo link mencionado anteriormente e o root_path indica o caminho do dataset caltech256 baixado para ser utilizado
    * split_ratio: esse decimal (0 a 1) indica a taxa de divisão do conjunto de dados de treinamento e de validação
    * savePath_idx: indica o caminho, no qual salva os índices das amostras para treinar, avaliar e testar o modelo. O Objetivo é quando for fazer o experimento                     utilizar o mesmo conjunto de teste com mesmas amostras, evitando usar, por ventura, imagens que tenham já sido vistas pelo modelo durante o treinamento. Esses ínfices serão salvos em um arquivo ".npy". Caso o caminho fornecido não exista, então o método separa os conjuntos de dados para treinar, validar e testar e salva os índices das amostras no arquivo ".npy". Caso contrário, o arquivo já existe no diretório fornecido, o método carrega os índices desses arquivos já existentes e utilizá-os


Variáveis declaradas no bloco de carregar os datasets:

model_name = "mobilenet"  # nome do modelo a ser treinado. A ideia é que na seu diretório-raiz tenha uma pasta para cada dataset e dentro dessa um diretório para cada modelo. 
dataset_name = "caltech256" # nome do dataset a ser utilizado para treinar, validar e testar o modelo
model_id = 1  # identifica o id do modelo, caso queira treinar e validar mais de um para testar. 
img_dim = 300 # dimensões (altura e largura) das imagens de entrada
input_dim = 300 # dimensões (altura e largura) das imagens de entrada
batch_size_train, batch_size_test = 32, 1 # define a quantidade de amostras contida em um lote (batch). Pode aumentar o lote de treinamento, contudo o de teste deve permanecer igual a 1.
split_ratio = 0.1 # esse decimal (0 a 1) indica a taxa de divisão do conjunto de dados de treinamento e de validação

Agora vou definir os diversos caminhos utilizados ao longo do arquivo.

root_dir = "..." #diretório-raiz
dataset_path = "..." #caminho em que está salvo o dataset Caltech 256

save_root_path = os.path.join(root_dir, dataset_name, model_name) # esse diretório que vai salvar todos os arquivores referentes ao treinamento do modelo. A ideia é que você tem um diretório-raix. Em seguida, separa-se os arquivos a ser salvo pelo modelo e pelo dataset utilizado.

model_save_path = os.path.join(save_root_path, "%s_%s_%s.pth"%(model_name, dataset_name, model_id)) # caminho no qual o modelo treinado será salvo.
history_save_path = os.path.join(save_root_path, "history_%s_%s_%s.pth"%(model_name, dataset_name, model_id)) #caminho que salva o desempenho obtido e o erro por epoch de treinamento e validação.

n_classes = 258 # número de classes existentes no dataset. Claro que se forem treinar usando o Cifar 10, vai ter que alterar.
pretrained = True # Se True, o seu modelo é inicializado com os pesos neurais pelo modelo treinado com o dataset ImageNet.
n_branches = 5 # Número de ramos laterais inseridos nas camadas intermediárias. Um dos experimentos é variar esse número
n_exits = n_branches + 1 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_shape = (3, input_dim, input_dim)
distribution = "linear" # insere os ramos laterais de fomra linear.
exit_type = "bnpool" # define o tipo de ramo inserido, insere uma camada BatchNormalization. Leia a sua função.
lr = [1.5e-4, 0.005] # define a taxa de aprendizado utilizada durante a otimização. Nesse caso, estou colocando como uma lista, pois estou definindo taxas de aprendizado diferente para os ramos laterais e para as cmadas do ramo principal.
weight_decay = 0.00005 

Os hiperparâmetros anteriores pode modificar para aprimorar o desempenho.


* Constroi o modelo de DNNs com saídas antecipadas
early_exit_model = Early_Exit_DNN(model_name, n_classes, pretrained, n_branches, input_shape, exit_type, device, distribution=distribution)
early_exit_model = early_exit_model.to(device)
early_exit_model.exits.to(device)

* Define a função de perda.
criterion = nn.CrossEntropyLoss()

* Note que diferentes partes do modelo utilizam taxas de aprendizado diferente.
optimizer = optim.Adam([{'params': early_exit_model.stages.parameters(), 'lr': lr[0]},
                       {'params': early_exit_model.exits.parameters(), 'lr': lr[1]},
                       {'params': early_exit_model.classifier.parameters(), 'lr': lr[1]}], weight_decay=weight_decay)

Contudo, vocês podem modificar para usar apenas uma taxa de aprendizado em todas da seguinte forma:
optimizer = optim.Adam([early_exit_model.parameters(), weight_decay=weight_decay)

Caso queiram usar o processo de otimização Stochastic Gradient Descendente basta procurar pro optim.SGD

* esse método adapta a taxa de aprendizado. Pode ser usado os seguintes
    * scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 10, eta_min=0, last_epoch=-1)

* https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.StepLR.html#torch.optim.lr_scheduler.StepLR

* permite atribuir pesos a cada um dos ramos. Um dos trabalhos da literatura utiliza esse esquema, porém podem testar usar np.ones(n_exits) também
loss_weights = np.linspace(0.15, 1, n_exits)

Por fim, note que o código é executado até o validation loss para de diminuir. Não é especificado um número de épocas, mas até o validation loss parar de diminuir .


Qualquer dúvida, avise-me. 

# Experimentos

Contexto: as redes neurais com saídas antecipadas para adecidir quais amostras devem ser classificadas antecipadamente calculam uma métrica de confiança para cada amostra e verifica se esse valor de confiança é maior do que um dado limiar (threshold)

Em relação aos experimentos, eu pensei em três experimetnos para fazer uma análise das redes neurais com saídas aantecipadas. 

Variar o número de saídas antecipadas de 2 a 5 e rodar experimento para verrificar quantas foram classificadas antecipadamente. Nesse experimento, é necessário treeinar modelos para cada quantidade de saída antecipada, porem aceito sugestoes e diferentes threshold.  

Além disso, seria legal um modelo que mostre o desempenho da rede neural para diferentes quantidade de ramos laterais e diferentes threshold

Por fim, uma vez terminado esses dois expeirmentos, vamos fazer os experimentos de tempo de inferência usando toda aquela parte em flask que ja estã no repositorio. 

A ideia é que o arquivo "end_node_img.py" envie imagens do conjunto de teste a borda. A borda recebe, execute o processamento até um predeterminado quantidade de ramos laterais e tente inferir na borda. Se classificar antecipadamente na boda, finaliza a medição do tempo. Caso não consiga, envie à nuvem que termine de executar a inferência e avisa a borda que terminou e termin a medição do tempo de inferência. Esse procedimento estoa nos seguintes arquivos. 

* /appEdge/api/controllers.py -> recebe a imagem chama a função de processamento que está no arquivo /appEdge/api/services/edgeProcessing.py .

Vamos nos comunicando.
