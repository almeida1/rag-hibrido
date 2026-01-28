package com.fatec.rag_hibrido;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

/**
 * O objetivo da biblioteca LangChain4j é simplificar a integração de LLMs em
 * aplicações Java.
 * O conhecimento do LLM é limitado aos dados com os quais ele foi treinado.
 * Se o projeto necessita que o LLM tenha conhecimento específico da área ou
 * dados proprietários, é possivel:
 * -Usar o método RAG (Geração Aumentada por Recuperação).
 * Ajuste fino do LLM com os dados.
 * Combinar o método RAG com o ajuste fino.
 * O processo RAG é dividido em duas etapas distintas: indexação e recuperação.
 * O LangChain4j fornece ferramentas para ambas as etapas.
 */
@SpringBootApplication
public class RagHibridoApplication {

	public static void main(String[] args) {
		SpringApplication.run(RagHibridoApplication.class, args);
	}

}
