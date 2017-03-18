"use strict";

/******************************************************************************
 * Perceptron Multicamadas (em Javascript).                                   *
 *                                                                            *
 * Implementação simples da rede Perceptron Multicamadas com o treinamento    *
 * utilizando o algoritmo "backpropagation" para classificação das amostras.  *
 *                                                                            *
 * Autor: Gilberto Augusto de Oliveira Bastos                                 *
 * Licença: BSD-2-Clause                                                      * 
 ******************************************************************************

/*************************
 * PeceptronMulticamadas *
 *************************/

/** Objeto que representa o Perceptron
 Multicamadas, armazenando as referências
 para as camadas da rede. */
function PerceptronMulticamadas()
{
    this.camadas = [];
}

/** Método que realiza a classificação de algum
 padrão apresentado à rede através do método 
 FeedFoward.
 
 Lembrando que o padrão (array) deve estar normalizado antes 
 de ser apresentado à rede para maior agilidade na classificação
 e treinamento. */
PerceptronMulticamadas.prototype.alimentarPerceptronFeedFoward =
    function (array){
	
	/* A primeira camada da rede é alimentada através
	 do array. */
	this.camadas[0].alimentarCamadaFeedFowardArray(array);
	
	/* As demais, são alimentadas pela ativação dos 
	 neurônios das camadas anteriores. */
	for (var c = 1; c < this.camadas.length; c++)
	{
	    this.camadas[c].alimentarCamadaFeedFoward(this.camadas[c - 1]);
	}
    };

/** Método que realiza o treinamento da rede neural através
 dos padrões de treinamento. 
 
 O método recebe a matriz ("matrix") que irá conter
 os padrões e o valor de saída desejado para cada padrão, 
 a taxa de aprendizagem e o erro desejado para que
 seja encerrado o treinamento ao Perceptron Multicamadas. 

 Exemplo de uma matriz de treinamento "OR" que irá conter
 os padrões para o treinamento:
 
 var matrizTreinamentoOR = [
   {padrao: [1, 0], objetivo: [1]},
   {padrao: [0, 1], objetivo: [1]},
   {padrao: [0, 0], objetivo: [0]},
   {padrao: [1, 1], objetivo: [1]}
];

 Lembrando que cada padrão de treinamento é um objeto, e os
 mesmos são agrupados através de um array convencional
 do Javascript. 

 A função irá retornar a quantidade de épocas necessárias para
 realizar o treinamento da rede neural. */
PerceptronMulticamadas.prototype.treinamentoBackProp =
    function(matrix, taxaAprendizagem, erroDesejado){

	/* Variável que vai armazenar o erro global da
	 rede após a apresentação dos padrões. */
	var erroGlobal;

	/* Variável que irá armazenar a quantidade de épocas
	 necessária para treinar a rede. */
	var epocas = 0;

	/* Realizando o treinamento... */
	do
	{
	    erroGlobal = 0.0;
	    /* Apresentando os padrões de treinamento para a rede e
	     realizando o treinamento da mesma. */
	    for (var i = 0; i < matrix.length; i++)
	    {
		/* Alimentando a rede com o padrão. */
		this.alimentarPerceptronFeedFoward(matrix[i].padrao);
		
		/* Realizando a retropropagação do erro para a última camada e
		 já somando o erro calculado para o padrão no erro global. */
		erroGlobal += 0.5 * this.camadas[this.camadas.length - 1].
		    calcularErroRetropropBackPropArray(matrix[i].objetivo);

		/* Realizando a retropropagaão do erro para as demais camadas. */
		for (var c = this.camadas.length - 2; c >= 0; c--)
		{
		    this.camadas[c].calcularErroRetropropBackProp(this.camadas[c + 1]);
		}

		/* Atualizando os pesos dos neurônios da primeira camada. */
		this.camadas[0].atualizarPesosNeuroniosArray(matrix[i].padrao, 
                    taxaAprendizagem);
		
		/* Atualizando os pesos dos neurônios das demais camadas. */
		for (c = 1; c < this.camadas.length; c++)
		{
		    this.camadas[c].atualizarPesosNeuroniosCamada(this.camadas[c - 1],
                        taxaAprendizagem);
		}

		/* Realizando o cálculo do MSE. */
		erroGlobal = erroGlobal / matrix.length;

		/* Atualizando a quantida de épocas. */
		epocas++;

		/* Imprimindo o erro MSE para a época atual. */
		console.log("Época: ", epocas);
		console.log("Erro MSE: ", erroGlobal);
	    }
	    
	} while (erroGlobal > erroDesejado);

	return epocas;
    };

/** Método que adiciona uma camada à rede Perceptron
 Multicamadas. */
PerceptronMulticamadas.prototype.adicionarCamada =
    function(camada) {
	this.camadas.push(camada);
    };

/**********
 * Camada *
 **********/

/** Objeto que irá representar uma 
 camada da rede neural FeedFoward.

 O construtor recebe a quantidade de neurônios
 da camada anterior a que está sendo criada no momento,
 e a quantidade de neurônios que a camada que está sendo criada
 deverá possuir, as funções de ativação e derivada da
 função de ativação e o intervalo para geração dos pesos
 dos neurônios. 

 Obs: Se esta for a primeira camada da rede, o parâmetro
 "qtdNeuroniosCamadaAnterior" deve ser igual a quantidade de
 itens (ou neurônios) da camada de entrada da rede. */
function Camada(qtdNeuroniosCamadaAnterior,
		qtdNeuroniosCamada,
		funcaoAtivacao,
		derivadaFuncaoAtivacao,
		min,max)
{
    this.neuronios = [];
    this.funcaoAtivacao = funcaoAtivacao;
    this.derivadaFuncaoAtivacao = derivadaFuncaoAtivacao;
    
    /* Criando os neurônios para esta 
     camada. */
    for (var i = 0; i < qtdNeuroniosCamada; i++)
    {
	this.neuronios.push(new Neuronio(qtdNeuroniosCamadaAnterior));
    }
}

/** Método que realiza a alimentação desta camada
 através do método FeedFoward.

 O método recebe a camada anterior à camada da qualn
 o método está sendo invocado e realiza a alimentação
 da rede. */
Camada.prototype.alimentarCamadaFeedFoward =
    function(camadaAnterior) {
	/* Percorrendo os neurônios da camada. */
	for (var n = 0; n < this.neuronios.length; n++)
	{
	    /* Calculando o valor da função de integração para
	     o neurônio "n-ésimo". */
	    var valorFuncaoIntegracao = 0.0;

	    for (var i = 0; i < camadaAnterior.neuronios.length; i++)
	    {
		valorFuncaoIntegracao += this.neuronios[n].w[i]
		    * camadaAnterior.neuronios[i].ativacao;
	    }

	    /* Calculando a ativação do neurônio com o uso do bias
	     com a sua derivada. */
	    this.neuronios[n].ativacao = this.funcaoAtivacao(valorFuncaoIntegracao +
							    this.neuronios[n].bias);
	    this.neuronios[n].derivadaAtivacao =
		this.derivadaFuncaoAtivacao(this.neuronios[n].ativacao);
	}
    };

/** Método que realiza a alimentação desta camada
 através do método FeedFoward.

 O método em vez de receber a camada anterior à esta,
 recebe um vetor contendo algum padrão para ser apresentado
 à camada. Usar este método para apresentar os padrões que 
 desejam ser classificados. */
Camada.prototype.alimentarCamadaFeedFowardArray =
    function(array) {
	/* Percorrendo os neurônios da camada. */
	for (var n = 0; n < this.neuronios.length; n++)
	{
	    /* Calculando o valor da função de integração para
	     o neurônio "n-ésimo". */
	    var valorFuncaoIntegracao = 0.0;

	    for (var i = 0; i < array.length; i++)
	    {
		valorFuncaoIntegracao +=
		    this.neuronios[n].w[i] * array[i];
	    }

	    /* Calculando a ativação do neurônio com o uso do bias
	     com a sua derivada. */
	    this.neuronios[n].ativacao = this.funcaoAtivacao(valorFuncaoIntegracao +
							    this.neuronios[n].bias);
	    this.neuronios[n].derivadaAtivacao =
		this.derivadaFuncaoAtivacao(this.neuronios[n].ativacao);
	}
    };

/** Método que realiza o cálculo dos erros retropropagados
 dos neurônios desta camada através do método Backprop.

 O método recebe a camada posterior à camada da qual 
 o método está sendo invocado e realiza o cálculo
 do erro retropropagado dos neurônios. */
Camada.prototype.calcularErroRetropropBackProp =
    function(camadaPosterior) {
	/* Percorrendo os neurônios da camada. */
	for (var n = 0; n < this.neuronios.length; n++)
	{
	    /* Calculando a soma dos erros da camada posterior

	     multiplicados pelos pesos do neurônio "n-ésimo". */
	    var somaErroCamPosterior = 0.0;

	    for (var i = 0; i < camadaPosterior.neuronios.length; i++)
	    {
		/* Calculando o erro do neurônio "i-ésimo" da camada posterior
		 multiplicado pelo respectivo peso da camada posterior que se
		 conecta ao respectivo neurônio "n-ésimo" que está tendo seu erro
		 calculado, e somando... */
		somaErroCamPosterior += camadaPosterior.neuronios[i].w[n] *
		    camadaPosterior.neuronios[i].erroRetroprop;
	    }

	    /* Por fim, calculando o erro retropropagado do neurônio. */
	    this.neuronios[n].erroRetroprop = this.neuronios[n].derivadaAtivacao *
		somaErroCamPosterior;
	}
    };

/** Método que realiza o cálculo dos erros retropropagados
 dos neurônios desta camada através do método Backprop.

 Em vez de receber a camada posterior, esse método recebe
 o padrão de saída desejado para esta camada através de
 um array, ou seja, essa função só deve ser executada
 se esta camada for a última camada do Perceptron 
 Multicamadas (camada de saída).

 A função retorna o erro do padrão apresentado à
 rede. */
Camada.prototype.calcularErroRetropropBackPropArray =
    function(array) {

	/* Variável que irá armazenar o erro para o padrão apresentado
	 à rede através do "array". */
	var erroPadrao = 0.0;

	/* Percorrendo os neurônios da camada. */
	for (var n = 0; n < this.neuronios.length; n++)
	{
	    /* Calculando do erro da saída para o neurônio "n-ésimo". */
	    var erroSaidaNeuronio = this.neuronios[n].ativacao -
		    array[n];

	    /* Calculando o erro retropropagado. */
	    this.neuronios[n].erroRetroprop = erroSaidaNeuronio *
		this.neuronios[n].derivadaAtivacao;

	    /* Calculando o erro para o padrão... */
	    erroPadrao += Math.pow(erroSaidaNeuronio, 2);
 	}

	/* Retornando o erro do padrão apresentado à rede. */
	return erroPadrao;
    };

/** Método que realiza a atualização dos pesos dos neurônios da
 camada. 

 O método recebe a camada anterior à esta camada e recebe também
 a taxa de aprendizagem para atualização dos pesos. */
Camada.prototype.atualizarPesosNeuroniosCamada =
    function(camadaAnterior, taxaAprendizagem){
	/* Percorrendo os neurônios da camada. */
	for (var n = 0; n < this.neuronios.length; n++)
	{
	    /* Percorrendo os pesos do neurônio "n-ésimo". */
	    for (var i = 0; i < camadaAnterior.neuronios.length; i++)
	    {
		/* Atualizando o peso "i-ésimo" do neurônio "n-ésimo". */
		this.neuronios[n].w[i] += -taxaAprendizagem *
		    camadaAnterior.neuronios[i].ativacao *
		    this.neuronios[n].erroRetroprop;
	    }

	    /* Atualizando o bias do neurônio "n-ésimo". */
	    this.neuronios[n].bias += -taxaAprendizagem *
		this.neuronios[n].erroRetroprop;
	}
    };

/** Método que realiza a atualização dos pesos dos neurônios da 
 camada.

 O método em vez de receber a camada anterior à esta, recebe um
 array contendo o padrão de entrada que se deseja apresentar
 à rede para ser classificado, além da taxa de aprendizagem. */
Camada.prototype.atualizarPesosNeuroniosArray =
    function(array, taxaAprendizagem){
	/* Percorrendo os neurônios da camada. */
	for (var n = 0; n < this.neuronios.length; n++)
	{
	    /* Percorrendo os pesos do neurônio "n-ésimo". */
	    for (var i = 0; i < array.length; i++)
	    {
		/* Atualizando o peso "i-ésimo" do neurônio "n-ésimo". */
		this.neuronios[n].w[i] += -taxaAprendizagem * array[i] *
		    this.neuronios[n].erroRetroprop;
	    }

	    /* Atualizando o bias do neurônio "n-ésimo"... */
	    this.neuronios[n].bias += -taxaAprendizagem * 
                    this.neuronios[n].erroRetroprop;
	}
    };

/************
 * Neurônio *
 ************

/** Objeto que irá representar um 
 neurônio de uma camada da rede neural
 FeedFoward.
 
 O construtor recebe a quantidade de pesos 
 que o neurônio deverá ter e o bias. Após isso, 
 serão gerados os pesos para o neurônio automáticamente
 no intervalo de "min" até "max" (se os parâmetros
 não forem informados, os pesos serão gerados no intervalo
 de -0.5 até 0.5.

 Obs: Caso o parâmetro bias não seja informado,
 o valor padrão para o mesmo será 1. */
function Neuronio(qtdPesosNeuronio, bias, min, max)
{
    this.ativacao = 0;
    this.derivadaAtivacao = 0;
    this.erroRetroprop = 0;
    bias === undefined ? this.bias = 1 : this.bias = bias;

    /* Verificando se os parâmetros para o intervalo dos
     pesos foram informados. */
    if (min == undefined || max == undefined)
    {
	/* Caso não tenham sido informados, atribuindo 
	 o valor padrão para os mesmos. */
	min = -0.5;
	max = 0.5;
    }
    
    /* Pesos deste neurônio. */
    this.w = [];

    /* Gerando os pesos aleatórios para o neurônio. */
    for (var i = 0; i < qtdPesosNeuronio; i++)
    {
	this.w.push(Math.random() * (max - min) + min);
    }
}

/***********************
 * Funções de ativação *
 ***********************/

/** Objeto que irá armazenar as funções de ativação
 e derivada da rede. */
var funcoesAtivacao = {};

funcoesAtivacao.degrau =
    function(z) {
	return (z >= 0) ? 1 : 0;
    };

funcoesAtivacao.derivadaDegrau =
    function(valDegrau) {
	return 1.0;
    };

funcoesAtivacao.sigmoide =
    function(z) {
	return 1.0 / ((1.0) + Math.exp(-z));
    };

funcoesAtivacao.derivadaSigmoide =
    function(valSig) {
	return valSig * (1.0 - valSig);
    };

funcoesAtivacao.tangHiperbolica =
    function(z) {
	return Math.tanh(z);
    };

funcoesAtivacao.derivadaTangHiperbolica =
    function(valTangHiperbolica) {
	return 1.0 - Math.pow(valTangHiperbolica, 2);
    };

/** Instanciando o Perceptron Multicamadas (2-2-1) e adicionando as
 camadas ao mesmo (de processamento e a camada de saída). */
var pm = new PerceptronMulticamadas();
pm.adicionarCamada(new Camada(2, 2, funcoesAtivacao.sigmoide,
			      funcoesAtivacao.derivadaSigmoide));
pm.adicionarCamada(new Camada(2, 1, funcoesAtivacao.sigmoide,
			      funcoesAtivacao.derivadaSigmoide));

/* Matriz para o treinamento da rede. */
var matrizTreinamentoXOR = [
    {padrao: [1, 0], objetivo: [1]},
    {padrao: [0, 1], objetivo: [1]},
    {padrao: [1, 1], objetivo: [0]},
    {padrao: [0, 0], objetivo: [0]}
];

/* Realizando o treinamento da rede com a matriz acima, 
 até que o erro da mesma seja menor ou igual a 0.005, com
 taxa de aprendizagem 0.3. */
pm.treinamentoBackProp(matrizTreinamentoXOR, 0.3, 0.001);

/* Realizando a alimentação da rede e imprimindo o resultado. */
console.log("\nResultados do treinamento (XOR):");
pm.alimentarPerceptronFeedFoward([0, 0]);
console.log("[0,0] -> ", pm.camadas[1].neuronios[0].ativacao);
pm.alimentarPerceptronFeedFoward([1, 0]);
console.log("[1,0] -> ", pm.camadas[1].neuronios[0].ativacao);
pm.alimentarPerceptronFeedFoward([0, 1]);
console.log("[0,1] -> ", pm.camadas[1].neuronios[0].ativacao);
pm.alimentarPerceptronFeedFoward([1, 1]);
console.log("[1,1] -> ", pm.camadas[1].neuronios[0].ativacao);
