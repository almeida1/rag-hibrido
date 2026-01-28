package com.fatec.rag_hibrido;

import org.junit.jupiter.api.Test;

import com.fatec.rag_hibrido.service.HybridRAGSystem;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.Metadata;
import dev.langchain4j.data.segment.TextSegment;

import static org.junit.jupiter.api.Assertions.*;
import java.util.Arrays;
import java.util.List;

/**
 * Objetivo - Testes unitários e de integração para o sistema HybridRAGSystem.
 * Esta classe valida o comportamento da busca híbrida e do algoritmo de fusão
 * de resultados, garantindo que o sistema recupere as informações corretas.
 */
public class HybridRAGSystemTest {
        /**
         * Objetivo - Validar a funcionalidade de recuperação híbrida processada
         * localmente
         * pelas bibliotecas LangChain4J e Lucene (mecanismo de busca híbrida).
         * Nao valida a resposta do modelo de linguagem ampla (neste exemplo Ollama)
         * Testa se o sistema consegue ingerir documentos e retornar o número correto
         * de resultados relevantes para uma consulta específica, combinando os pesos
         * de BM25 e Embeddings. O sistema consegue encontrar os documentos corretos
         * no banco de dados local da memória.
         */
        @Test
        public void ct01_req01_testRetrieveHybrid() {
                HybridRAGSystem ragSystem = new HybridRAGSystem("ollama", "qwen3:4b", "demo", "http://127.0.0.1:11434",
                                1);
                List<Document> documents = Arrays.asList(
                                Document.from("A inteligência artificial está transformando a medicina.",
                                                Metadata.from("fonte", "artigo_ciencia").put("ano", "2023")),
                                Document.from("Machine learning é um subcampo da IA.",
                                                Metadata.from("fonte", "wiki").put("ano", "2022")),
                                Document.from("Deep learning usa redes neurais profundas.",
                                                Metadata.from("fonte", "livro").put("ano", "2023")),
                                Document.from("Brasil é o maior país da América do Sul.",
                                                Metadata.from("fonte", "geografia").put("ano", "2024")),
                                Document.from("Python é popular para ciência de dados.",
                                                Metadata.from("fonte", "programacao").put("ano", "2023")));
                ragSystem.loadDocuments(documents);
                List<TextSegment> resultados = ragSystem.retrieveHybrid(
                                "O que é inteligência artificial?",
                                3, // maxResults
                                0.5, // bm25Weight
                                0.5 // embeddingWeight
                );
                assertEquals(3, resultados.size());
                // teste E2E
                // String resposta = ragSystem.answer("O que é inteligência artificial?");
                // assertNotNull(resposta);
        }

        /**
         * Objetivo - Testar a lógica de fusão de rankings (RRF).
         * Verifica se o sistema consegue processar documentos carregados e realizar
         * a fusão de resultados de forma consistente, garantindo que a lista final
         * de segmentos não esteja vazia e contenha dados válidos.
         */
        @Test
        void ct02_req02_testReciprocalRankFusion() {
                // Testar lógica de fusão
                HybridRAGSystem rag = new HybridRAGSystem("ollama", "qwen3:4b", "demo", "http://127.0.0.1:11434", 1);

                // Para o teste funcionar, precisamos carregar os documentos no sistema
                TextSegment seg1 = TextSegment.from("Doc1");
                TextSegment seg2 = TextSegment.from("Doc2");

                rag.loadDocuments(Arrays.asList(
                                dev.langchain4j.data.document.Document.from(seg1.text()),
                                dev.langchain4j.data.document.Document.from(seg2.text())));

                // Deve combinar ambos os resultados
                List<TextSegment> fused = rag.retrieveHybrid("Doc1", 2, 0.5, 0.5);
                assertFalse(fused.isEmpty());
                assertTrue(fused.get(0).text().contains("Doc1") || fused.get(0).text().contains("Doc2"));
        }
}
