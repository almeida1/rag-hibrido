package com.fatec.rag_hibrido.service;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.splitter.DocumentByParagraphSplitter;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.bge.small.en.v15.BgeSmallEnV15EmbeddingModel;
import dev.langchain4j.model.openai.OpenAiEmbeddingModel;
import dev.langchain4j.store.embedding.EmbeddingMatch;

import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import java.util.*;

public class HybridRAGSystem {
    private final EmbeddingModel embeddingModel;
    private final EmbeddingStore<TextSegment> embeddingStore;
    private final BM25Retriever bm25Retriever;
    private final DocumentSplitter splitter;

    public HybridRAGSystem() {
        this.embeddingModel = new BgeSmallEnV15EmbeddingModel();
        this.embeddingStore = new InMemoryEmbeddingStore<>();
        this.bm25Retriever = new BM25Retriever();
        this.splitter = new DocumentByParagraphSplitter(500, 50);
    }

    // Construtor com OpenAI
    public HybridRAGSystem(String openAiApiKey) {
        this.embeddingModel = OpenAiEmbeddingModel.builder()
                .apiKey(openAiApiKey)
                .modelName("text-embedding-3-small")
                .build();
        this.embeddingStore = new InMemoryEmbeddingStore<>();
        this.bm25Retriever = new BM25Retriever();
        this.splitter = new DocumentByParagraphSplitter(500, 50);
    }

    public void loadDocuments(List<Document> documents) {
        for (Document doc : documents) {
            // Dividir documento em segmentos
            List<TextSegment> segments = splitter.split(doc);

            for (TextSegment segment : segments) {
                // Adicionar ao BM25
                bm25Retriever.addDocument(segment);

                // Adicionar ao embedding store
                Embedding embedding = embeddingModel.embed(segment).content();
                embeddingStore.add(embedding, segment);
            }
        }
        System.out.println("Documentos carregados: " + documents.size());
    }

    public List<TextSegment> retrieveHybrid(String query, int maxResults,
            double bm25Weight, double embeddingWeight) {
        // Recuperar usando BM25
        List<TextSegment> bm25Results = bm25Retriever.retrieve(query, maxResults * 2);

        // Recuperar usando embeddings
        Embedding queryEmbedding = embeddingModel.embed(query).content();
        List<EmbeddingMatch<TextSegment>> embeddingResults = embeddingStore.findRelevant(queryEmbedding,
                maxResults * 2);

        // Combinar resultados usando RRF (Reciprocal Rank Fusion)
        return reciprocalRankFusion(bm25Results, embeddingResults, maxResults);
    }

    /**
     * Objetivo - Reciprocal Rank Fusion (RRF) - Diferente de um RAG simples que
     * apenas busca e entrega, esta aplicação implementa uma camada de
     * inteligência na combinação dos resultados
     * O RRF é uma técnica sofisticada que não depende da escala dos scores (que são
     * diferentes no BM25 e no Cosseno dos Embeddings)
     * para ordenar os resultados, garantindo que documentos que aparecem bem
     * posicionados
     * em ambos os métodos subam para o topo da lista final.
     * 
     * @param bm25Results
     * @param embeddingResults
     * @param maxResults
     * @return
     */
    private List<TextSegment> reciprocalRankFusion(
            List<TextSegment> bm25Results,
            List<EmbeddingMatch<TextSegment>> embeddingResults,
            int maxResults) {

        Map<String, Double> scores = new HashMap<>();
        Map<String, TextSegment> allSegments = new HashMap<>();

        final double k = 60.0; // Constante de suavização

        // Processar resultados BM25
        for (int rank = 0; rank < bm25Results.size(); rank++) {
            TextSegment segment = bm25Results.get(rank);
            String contentHash = Integer.toHexString(segment.text().hashCode());

            allSegments.putIfAbsent(contentHash, segment);

            double score = scores.getOrDefault(contentHash, 0.0);
            score += 1.0 / (rank + k);
            scores.put(contentHash, score);
        }

        // Processar resultados de embeddings
        for (int rank = 0; rank < embeddingResults.size(); rank++) {
            TextSegment segment = embeddingResults.get(rank).embedded();
            String contentHash = Integer.toHexString(segment.text().hashCode());

            allSegments.putIfAbsent(contentHash, segment);

            double score = scores.getOrDefault(contentHash, 0.0);
            score += 1.0 / (rank + k);
            scores.put(contentHash, score);
        }

        // Ordenar por score
        List<Map.Entry<String, Double>> sortedEntries = new ArrayList<>(scores.entrySet());
        sortedEntries.sort((a, b) -> Double.compare(b.getValue(), a.getValue()));

        // Coletar resultados
        List<TextSegment> results = new ArrayList<>();
        for (int i = 0; i < Math.min(maxResults, sortedEntries.size()); i++) {
            String contentHash = sortedEntries.get(i).getKey();
            results.add(allSegments.get(contentHash));
        }

        return results;
    }

    // Método de fusão linear alternativa
    private List<TextSegment> linearFusion(
            List<TextSegment> bm25Results,
            List<EmbeddingMatch<TextSegment>> embeddingResults,
            int maxResults,
            double bm25Weight,
            double embeddingWeight) {

        Map<String, FusionScore> scores = new HashMap<>();

        // Normalizar scores BM25
        double maxBm25Score = bm25Results.stream()
                .mapToDouble(s -> Double.parseDouble(
                        s.metadata().toMap().getOrDefault("bm25_score", "0.0").toString()))
                .max()
                .orElse(1.0);

        for (TextSegment segment : bm25Results) {
            String id = segment.metadata().toMap().getOrDefault("id", segment.text()).toString();
            double bm25Score = Double.parseDouble(
                    segment.metadata().toMap().getOrDefault("bm25_score", "0.0").toString()) / maxBm25Score;

            scores.put(id, new FusionScore(bm25Score * bm25Weight, 0.0, segment));
        }

        // Processar embeddings (já vêm com scores normalizados)
        for (EmbeddingMatch<TextSegment> result : embeddingResults) {
            TextSegment segment = result.embedded();
            String id = segment.metadata().toMap().getOrDefault("id", segment.text()).toString();

            double embeddingScore = result.score();
            FusionScore fusionScore = scores.getOrDefault(id,
                    new FusionScore(0.0, 0.0, segment));

            fusionScore.embeddingScore = embeddingScore * embeddingWeight;
            fusionScore.segment = segment;
            scores.put(id, fusionScore);
        }

        // Calcular score combinado e ordenar
        List<Map.Entry<String, FusionScore>> sorted = new ArrayList<>(scores.entrySet());
        sorted.sort((a, b) -> Double.compare(
                b.getValue().totalScore(),
                a.getValue().totalScore()));

        return sorted.stream()
                .limit(maxResults)
                .map(entry -> entry.getValue().segment)
                .collect(java.util.stream.Collectors.toList());
    }

    private static class FusionScore {
        double bm25Score;
        double embeddingScore;
        TextSegment segment;

        FusionScore(double bm25Score, double embeddingScore, TextSegment segment) {
            this.bm25Score = bm25Score;
            this.embeddingScore = embeddingScore;
            this.segment = segment;
        }

        double totalScore() {
            return bm25Score + embeddingScore;
        }
    }
}
