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












