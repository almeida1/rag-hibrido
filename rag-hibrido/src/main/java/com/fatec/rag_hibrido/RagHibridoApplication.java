package com.fatec.rag_hibrido;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

/**
 * RAG Hibrido - Arquitetura: Bi-encoder (para embeddings) + Inverted Index
 * (BM25 - Best Match 25).
 * Uso Ideal: Pesquisas em documentos técnicos, jurídicos ou científicos onde
 * tanto o contexto quanto as palavras exatas são críticos para a resposta
 * correta.
 * A arquitetura do sistema resolve os dois problemas principais do RAG ingênuo:
 * 1. Baixa Precisão em Termos Específicos: Resolvido pelo BM25.
 * 2. Perda de Contexto Semântico: Resolvido pelos Embeddings
 * O sistema usa o BGE Small (local) para criar os índices de busca e o Llama 3
 * (local via Ollama) para gerar as respostas finais.
 */
@SpringBootApplication
public class RagHibridoApplication {

	public static void main(String[] args) {
		SpringApplication.run(RagHibridoApplication.class, args);
	}

}
