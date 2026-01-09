package com.fatec.rag_hibrido.service;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.Metadata;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.openai.OpenAiChatModel;

import java.util.Arrays;
import java.util.List;

public class ExemploRAGHibrido {
    public static void main(String[] args) {
        // 1. Criar sistema RAG
        HybridRAGSystem ragSystem = new HybridRAGSystem();

        // 2. Criar documentos de exemplo
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

        // 3. Carregar documentos no sistema
        ragSystem.loadDocuments(documents);

        // 4. Testar busca híbrida
        List<String> consultas = Arrays.asList(
                "O que é inteligência artificial?",
                "Informações sobre Brasil",
                "Aprendizado de máquina");
        for (String consulta : consultas) {
            System.out.println("\n" + "=".repeat(50));
            System.out.println("Consulta: " + consulta);
            System.out.println("=".repeat(50));
            List<TextSegment> resultados = ragSystem.retrieveHybrid(
                    consulta,
                    3, // maxResults
                    0.5, // bm25Weight
                    0.5 // embeddingWeight
            );
            // 5. Sistema com geração de resposta
            testarRAGComGeracao();
            for (int i = 0; i < resultados.size(); i++) {
                TextSegment segment = resultados.get(i);
                System.out.println("\nResultado " + (i + 1) + ":");
                System.out.println("Texto: " + segment.text().substring(0,
                        Math.min(100, segment.text().length())) + "...");
                System.out.println("Metadata: " + segment.metadata().asMap());
            }
        }
    }

    public static void testarRAGComGeracao() {
        System.out.println("\n" + "=".repeat(50));
        System.out.println("Sistema RAG Completo com Geração");
        System.out.println("=".repeat(50));

        // Usando OpenAI para embeddings e geração
        String openAiApiKey = "demo"; // Configure sua chave

        HybridRAGSystem ragSystem = new HybridRAGSystem(openAiApiKey);

        // Configurar LLM para geração
        OpenAiChatModel llm = OpenAiChatModel.builder()
                .apiKey(openAiApiKey)
                .modelName("gpt-4o-mini")
                .temperature(0.0)
                .build();

        // Carregar documentos
        List<Document> docs = Arrays.asList(
                Document.from("LangChain4j é um framework Java para LLMs."),
                Document.from("RAG combina recuperação com geração de texto."),
                Document.from("BM25 é um algoritmo de recuperação baseado em frequência."));

        ragSystem.loadDocuments(docs);

        // Consulta
        // String consulta = "Explique o que é RAG";
        String consulta = "O que é Machine Learning";
        // Recuperar contexto
        List<TextSegment> contextos = ragSystem.retrieveHybrid(consulta, 2, 0.5, 0.5);

        // Construir prompt com contexto
        StringBuilder contextBuilder = new StringBuilder();
        for (TextSegment ctx : contextos) {
            contextBuilder.append(ctx.text()).append("\n\n");
        }

        String prompt = String.format(
                "Com base nos seguintes documentos:\n\n%s\n\nResponda: %s",
                contextBuilder.toString(),
                consulta);

        // Gerar resposta
        String resposta = llm.generate(prompt);
        System.out.println(">>>Resposta: " + resposta);
    }
}
