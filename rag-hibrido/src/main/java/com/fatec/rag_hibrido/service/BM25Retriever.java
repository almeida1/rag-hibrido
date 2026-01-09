package com.fatec.rag_hibrido.service;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;

import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.StringField;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.*;
import org.apache.lucene.queryparser.classic.QueryParser;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.similarities.BM25Similarity;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.Directory;

import dev.langchain4j.data.document.Metadata;
import dev.langchain4j.data.segment.TextSegment;

public class BM25Retriever {
    private final Directory directory;
    private final IndexWriter writer;
    private final Analyzer analyzer;
    private final Map<String, TextSegment> idToSegment;

    public BM25Retriever() {
        try {
            this.directory = new ByteBuffersDirectory();
            this.analyzer = new StandardAnalyzer();

            IndexWriterConfig config = new IndexWriterConfig(analyzer);
            config.setOpenMode(IndexWriterConfig.OpenMode.CREATE_OR_APPEND);
            this.writer = new IndexWriter(directory, config);

            this.idToSegment = new HashMap<>();
        } catch (Exception e) {
            throw new RuntimeException("Erro ao inicializar BM25Retriever", e);
        }
    }

    public void addDocument(TextSegment segment) {
        try {
            String id = UUID.randomUUID().toString();
            idToSegment.put(id, segment);

            Document doc = new Document();
            doc.add(new StringField("id", id, Field.Store.YES));
            doc.add(new TextField("content", segment.text(), Field.Store.YES));

            // Adicionar metadados
            if (segment.metadata() != null) {
                for (var entry : segment.metadata().toMap().entrySet()) {
                    doc.add(new StringField(
                            "meta_" + entry.getKey(),
                            entry.getValue().toString(),
                            Field.Store.YES));
                }
            }
            writer.addDocument(doc);
            writer.commit();
        } catch (Exception e) {
            throw new RuntimeException("Erro ao indexar documento", e);
        }
    }

    public List<TextSegment> retrieve(String query, int maxResults) {
        try {
            IndexReader reader = DirectoryReader.open(directory);
            IndexSearcher searcher = new IndexSearcher(reader);

            // Configurar BM25 (padrão do Lucene)
            searcher.setSimilarity(new BM25Similarity());

            // Criar query
            QueryParser parser = new QueryParser("content", analyzer);
            org.apache.lucene.search.Query luceneQuery = parser.parse(QueryParser.escape(query));

            // Executar busca
            TopDocs topDocs = searcher.search(luceneQuery, maxResults);

            List<TextSegment> results = new ArrayList<>();
            for (ScoreDoc scoreDoc : topDocs.scoreDocs) {
                Document doc = searcher.storedFields().document(scoreDoc.doc);
                String id = doc.get("id");
                TextSegment segment = idToSegment.get(id);

                // Clonar segment com score
                Map<String, Object> metadataMap = new HashMap<>(segment.metadata().toMap());
                metadataMap.put("bm25_score", scoreDoc.score);

                results.add(TextSegment.from(segment.text(), Metadata.from(metadataMap)));
            }

            reader.close();
            return results;
        } catch (Exception e) {
            throw new RuntimeException("Erro na recuperação BM25", e);
        }

    }

    public void close() {
        try {
            writer.close();
            directory.close();
        } catch (Exception e) {
            // Ignorar erros no fechamento
        }
    }
}