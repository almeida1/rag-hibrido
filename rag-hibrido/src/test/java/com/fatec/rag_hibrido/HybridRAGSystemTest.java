package com.fatec.rag_hibrido;

import org.junit.jupiter.api.Test;

import com.fatec.rag_hibrido.service.HybridRAGSystem;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.Metadata;
import dev.langchain4j.data.segment.TextSegment;

import static org.junit.jupiter.api.Assertions.*;
import java.util.Arrays;
import java.util.List;

public class HybridRAGSystemTest {
    @Test
    public void testRetrieveHybrid() {
        HybridRAGSystem ragSystem = new HybridRAGSystem();
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
    }

    @Test
    void testReciprocalRankFusion() {
        // Testar lógica de fusão
        HybridRAGSystem rag = new HybridRAGSystem();

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
