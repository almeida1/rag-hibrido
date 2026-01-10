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

import dev.langchain4j.store.embedding.EmbeddingSearchRequest;
import dev.langchain4j.store.embedding.EmbeddingSearchResult;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.model.chat.ChatLanguageModel;
import dev.langchain4j.model.openai.OpenAiChatModel;
import dev.langchain4j.model.ollama.OllamaChatModel;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import jakarta.annotation.PreDestroy;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import java.util.*;

@Service
public class HybridRAGSystem {
    private final EmbeddingModel embeddingModel;
    private final EmbeddingStore<TextSegment> embeddingStore;
    private final BM25Retriever bm25Retriever;
    private final DocumentSplitter splitter;
    private final ChatLanguageModel chatModel;

    public HybridRAGSystem() {
        this("demo", "llama3");
    }

    /**
     * BGE (BAII General Embedding): Criado pela BAII (Beijing Academy of Artificial
     * Intelligence).
     * Small: Indica que é a versão "leve" ou compacta do modelo.
     * En: Significa English. O modelo foi otimizado para textos em inglês
     * 
     * @param openAiApiKey
     */
    public HybridRAGSystem(@Value("${langchain4j.open-ai.api-key:demo}") String openAiApiKey,
            @Value("${ollama.model.name:llama3}") String ollamaModelName) {
        if ("demo".equals(openAiApiKey) || openAiApiKey == null || openAiApiKey.isBlank()) {
            this.embeddingModel = new BgeSmallEnV15EmbeddingModel();
            this.chatModel = OllamaChatModel.builder()
                    .baseUrl("http://localhost:11434")
                    .modelName(ollamaModelName)
                    .temperature(0.0)
                    .build();
        } else {
            this.embeddingModel = OpenAiEmbeddingModel.builder()
                    .apiKey(openAiApiKey)
                    .modelName("text-embedding-3-small")
                    .build();
            this.chatModel = OpenAiChatModel.builder()
                    .apiKey(openAiApiKey)
                    .modelName("gpt-4o-mini")
                    .temperature(0.0)
                    .build();
        }
        this.embeddingStore = new InMemoryEmbeddingStore<>();
        this.bm25Retriever = new BM25Retriever();
        this.splitter = new DocumentByParagraphSplitter(500, 50);

        System.out.println("SISTEMA RAG INICIALIZADO:");
        System.out.println("- Embedding Model: "
                + (embeddingModel instanceof BgeSmallEnV15EmbeddingModel ? "BgeSmallEnV15 (Local)" : "OpenAI"));
        System.out.println("- Chat Model: "
                + (chatModel instanceof OllamaChatModel ? "Configurado (Ollama: " + ollamaModelName + ")"
                        : "Configurado (OpenAI)"));
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

    public String answer(String query) {
        // Obter contextos com threshold de relevância
        List<TextSegment> contexts = retrieveHybrid(query, 5, 0.5, 0.5);

        // Se não houver contextos relevantes, responder que não sabe
        if (contexts.isEmpty()) {
            return "Desculpe, mas não encontrei informações nos documentos carregados para responder a essa pergunta com precisão.";
        }

        if (chatModel == null) {
            return "Modelo de Chat (LLM) não configurado. Para habilitar respostas completas, configure 'langchain4j.open-ai.api-key' no application.properties.\n\n"
                    +
                    "No entanto, encontrei " + contexts.size() + " trechos que podem ser relevantes nos documentos.";
        }

        StringBuilder contextBuilder = new StringBuilder();
        for (TextSegment ctx : contexts) {
            contextBuilder.append("- ").append(ctx.text()).append("\n\n");
        }

        String prompt = String.format(
                "Você é um assistente prestativo. Use APENAS os contextos abaixo para responder à pergunta.\n" +
                        "Se a resposta não estiver nos contextos, diga que não tem informações para responder.\n\n" +
                        "Contextos:\n%s\n\n" +
                        "Pergunta: %s\n\n" +
                        "Resposta:",
                contextBuilder.toString(),
                query);

        return chatModel.generate(prompt);
    }

    @PreDestroy
    public void close() {
        if (bm25Retriever != null) {
            bm25Retriever.close();
        }
    }

    public List<TextSegment> retrieveHybrid(String query, int maxResults,
            double bm25Weight, double embeddingWeight) {
        // Recuperar usando BM25
        List<TextSegment> bm25Results = bm25Retriever.retrieve(query, maxResults * 2);

        // Recuperar usando embeddings com threshold de similaridade
        Embedding queryEmbedding = embeddingModel.embed(query).content();
        EmbeddingSearchRequest searchRequest = EmbeddingSearchRequest.builder()
                .queryEmbedding(queryEmbedding)
                .maxResults(maxResults * 2)
                .minScore(0.65) // Threshold para evitar resultados totalmente irrelevantes
                .build();

        EmbeddingSearchResult<TextSegment> searchResult = embeddingStore.search(searchRequest);
        List<EmbeddingMatch<TextSegment>> embeddingResults = searchResult.matches();

        // Se nenhum método retornou nada decente, retorna lista vazia
        if (bm25Results.isEmpty() && embeddingResults.isEmpty()) {
            return Collections.emptyList();
        }

        // Combinar resultados usando RRF (Reciprocal Rank Fusion)
        return reciprocalRankFusion(bm25Results, embeddingResults, maxResults);
    }

    /**
     * Objetivo - Reciprocal Rank Fusion (RRF) - Diferente de um RAG simples que
     * apenas busca e entrega, esta aplicação implementa uma camada de
     * inteligência na combinação dos resultados
     * O RRF é uma técnica que não depende da escala dos scores (que são
     * diferentes no BM25 e no Cosseno dos Embeddings)
     * para ordenar os resultados, garantindo que documentos que aparecem bem
     * posicionados em ambos os métodos subam para o topo da lista final.
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
