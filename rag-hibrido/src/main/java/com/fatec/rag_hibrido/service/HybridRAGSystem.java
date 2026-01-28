package com.fatec.rag_hibrido.service;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.splitter.DocumentByParagraphSplitter;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.bge.small.en.v15.BgeSmallEnV15EmbeddingModel;
import dev.langchain4j.store.embedding.EmbeddingMatch;

import dev.langchain4j.store.embedding.EmbeddingSearchRequest;
import dev.langchain4j.store.embedding.EmbeddingSearchResult;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.model.chat.ChatLanguageModel;
import dev.langchain4j.model.openai.OpenAiChatModel;
import dev.langchain4j.model.ollama.OllamaChatModel;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import jakarta.annotation.PreDestroy;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import java.util.*;

/**
 * Objetivo - Sistema de RAG (Geração Aumentada por Recuperação) híbrido.
 * Esta classe gerencia a ingestão de documentos, a recuperação de contextos
 * usando busca híbrida (BM25 + Vetorial) e a geração de respostas
 * fundamentadas através de modelos de linguagem (LLM).
 * Ingestão em memória, não persistente, diminui o investimento em hardware.
 * 
 * @author Fatec
 * @version 1.0
 * @since 2026-01-24
 */
@Service
public class HybridRAGSystem {
    private final EmbeddingModel embeddingModel;
    private final EmbeddingStore<TextSegment> embeddingStore;
    private final BM25Retriever bm25Retriever;
    private final DocumentSplitter splitter;
    private final ChatLanguageModel chatModel;
    Logger logger = LogManager.getLogger(this.getClass());

    /**
     * Objetivo - Construtor do sistema RAG híbrido.
     * Inicializa os modelos de embedding e chat, configura os vetores
     * e os retrievers necessários para o funcionamento do sistema.
     * 
     * @param provider       O provedor do modelo de linguagem (openai ou ollama).
     * @param modelName      O nome do modelo a ser utilizado.
     * @param apiKey         A chave de API (para OpenAI).
     * @param baseUrl        A URL base do serviço (para Ollama).
     * @param timeoutSeconds O tempo máximo de espera para respostas do modelo.
     */
    public HybridRAGSystem(
            @Value("${rag.llm.provider:ollama}") String provider,
            @Value("${rag.llm.model-name:qwen3:4b}") String modelName,
            @Value("${rag.llm.api-key:demo}") String apiKey,
            @Value("${rag.llm.base-url:http://127.0.0.1:11434}") String baseUrl,
            @Value("${rag.llm.timeout-seconds:300}") Integer timeoutSeconds) {
        logger.info(">>>>>> HybridRAGSystem - Iniciado");
        this.embeddingModel = new BgeSmallEnV15EmbeddingModel(); // Embedding Model BGE Small open source

        if ("openai".equalsIgnoreCase(provider.trim())) {
            this.chatModel = OpenAiChatModel.builder()
                    .apiKey(apiKey)
                    .modelName(modelName)
                    .timeout(java.time.Duration.ofSeconds(timeoutSeconds))
                    .temperature(0.0)
                    .build();
            logger.info(">>>>>> HybridRAGSystem - Chat Model: OpenAI (" + modelName + ")");
        } else {
            this.chatModel = OllamaChatModel.builder()
                    .baseUrl(baseUrl)
                    .modelName(modelName)
                    .timeout(java.time.Duration.ofSeconds(timeoutSeconds))
                    .temperature(0.0)
                    .build();
            logger.info(">>>>>> HybridRAGSystem - Chat Model: Ollama (" + modelName + " em " + baseUrl + ")");
        }

        this.embeddingStore = new InMemoryEmbeddingStore<>();
        this.bm25Retriever = new BM25Retriever();
        this.splitter = new DocumentByParagraphSplitter(500, 50);
        logger.info(">>>>>> HybridRAGSystem - Embedding Model: BgeSmallEnV15 (Local)");
    }

    /**
     * Objetivo - Carregar e processar uma lista de documentos no sistema.
     * Divide os documentos em parágrafos, gera embeddings para cada segmento e
     * os indexa tanto no buscador BM25 quanto no mecanismo de armazenamento de
     * vetores.
     * 
     * @param documents Lista de documentos a serem ingeridos.
     */
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
        logger.info(">>>>>> HybridRAGSystem - Documentos carregados: " + documents.size());
    }

    /**
     * Objetivo - Responder a uma pergunta baseada nos documentos carregados.
     * Recupera os contextos mais relevantes através de busca híbrida e utiliza
     * o modelo de linguagem para gerar uma resposta fundamentada.
     * 
     * @param query A pergunta feita pelo usuário.
     * @return A resposta gerada pelo modelo ou uma mensagem de
     *         erro/indisponibilidade.
     */
    public String answer(String query) {
        // Obter contextos com threshold de relevância
        List<TextSegment> contexts = retrieveHybrid(query, 5, 0.5, 0.5);

        // Se não houver contextos relevantes, responder que não sabe
        if (contexts.isEmpty()) {
            return "Não tenho informações para responder";
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

        try {
            return chatModel.generate(prompt);
        } catch (Exception e) {
            String errorMessage = e.getMessage() != null ? e.getMessage() : "";
            if (errorMessage.contains("404") || errorMessage.toLowerCase().contains("not found")) {
                return "Erro: O modelo de linguagem (LLM) não foi encontrado ou não está disponível. " +
                        "Verifique se o Ollama está rodando e se o modelo está baixado (ollama pull "
                        + (chatModel instanceof OllamaChatModel ? "modelo" : "qwen3:4b") + "). Detalhes: "
                        + errorMessage;
            }
            return "Erro ao gerar resposta com o modelo de linguagem: " + errorMessage;
        }
    }

    /**
     * Objetivo - Liberar recursos antes da destruição do objeto.
     * Fecha o retriever BM25 para garantir que os arquivos de índice sejam
     * liberados corretamente.
     */
    @PreDestroy
    public void close() {
        if (bm25Retriever != null) {
            bm25Retriever.close();
        }
    }

    /**
     * Objetivo - Realizar uma busca híbrida combinando léxico (BM25) e semântico
     * (Embeddings).
     * Executa ambas as buscas e funde os resultados para obter o melhor de dois
     * mundos:
     * precisão de palavras-chave e compreensão de contexto.
     * 
     * @param query           A consulta de busca.
     * @param maxResults      Número máximo de resultados a retornar.
     * @param bm25Weight      Peso atribuído à busca léxica (usado em fusão linear).
     * @param embeddingWeight Peso atribuído à busca semântica (usado em fusão
     *                        linear).
     * @return Lista de segmentos de texto mais relevantes.
     */
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
     * Objetivo - Fusão de Rankings Recíprocos (RRF).
     * Combina os rankings de diferentes métodos de busca sem depender da escala
     * dos scores originais. Favorece documentos que aparecem bem posicionados
     * em múltiplas fontes de busca.
     * 
     * @param bm25Results      Resultados da busca léxica.
     * @param embeddingResults Resultados da busca por similaridade de vetores.
     * @param maxResults       Número de resultados finais desejados.
     * @return Lista consolidada e reordenada de segmentos.
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

    /**
     * Objetivo - Metodo de recuperacao alternativo - Fusão Linear de Scores.
     * Quando os pesos são iguais, a fusão linear é idêntica ao RRF.
     * Combina os resultados normalizando os scores originais e aplicando pesos
     * definidos pelo usuário para cada modalidade de busca.
     * O especialista de domínio pode ajustar os pesos para favorecer uma modalidade
     * de busca sobre a outra - em doc técnicos, com muitos codigos de erros ou
     * siglas especificas
     * por exemplo, pode-se favorecer a busca por semântica,
     * dar peso 0.8 para BM25 e 0.2 para Embeddings.
     * 
     * @param bm25Results      Resultados BM25.
     * @param embeddingResults Resultados de Embeddings.
     * @param maxResults       Limite de resultados.
     * @param bm25Weight       Peso da busca BM25.
     * @param embeddingWeight  Peso da busca de embeddings.
     * @return Lista de segmentos ordenada por score ponderado.
     */
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

    /**
     * Objetivo - Estrutura auxiliar para cálculo de fusão de scores.
     */
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
