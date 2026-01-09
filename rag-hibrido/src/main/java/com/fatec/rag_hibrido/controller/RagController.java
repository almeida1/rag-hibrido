package com.fatec.rag_hibrido.controller;

import com.fatec.rag_hibrido.dto.FolderIngestRequest;
import com.fatec.rag_hibrido.dto.IngestRequest;
import com.fatec.rag_hibrido.dto.QueryRequest;
import com.fatec.rag_hibrido.dto.QueryResponse;
import com.fatec.rag_hibrido.service.HybridRAGSystem;
import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.Metadata;
import dev.langchain4j.data.document.loader.FileSystemDocumentLoader;
import dev.langchain4j.data.document.parser.apache.pdfbox.ApachePdfBoxDocumentParser;
import dev.langchain4j.data.document.parser.apache.poi.ApachePoiDocumentParser;
import dev.langchain4j.data.document.parser.TextDocumentParser;
import dev.langchain4j.data.segment.TextSegment;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

@RestController
@RequestMapping("/api/rag")
public class RagController {

    private final HybridRAGSystem ragSystem;

    public RagController(HybridRAGSystem ragSystem) {
        this.ragSystem = ragSystem;
    }

    @PostMapping("/ingest")
    public ResponseEntity<String> ingest(@RequestBody IngestRequest request) {
        List<Document> documents = request.getDocuments().stream()
                .map(docDto -> {
                    Metadata metadata = docDto.getMetadata() != null ? Metadata.from(docDto.getMetadata())
                            : new Metadata();
                    return Document.from(docDto.getContent(), metadata);
                })
                .collect(Collectors.toList());

        ragSystem.loadDocuments(documents);
        return ResponseEntity.ok("Successfully ingested " + documents.size() + " documents.");
    }

    @PostMapping("/ingest/folder")
    public ResponseEntity<String> ingestFolder(@RequestBody FolderIngestRequest request) {
        try {
            List<Document> documents = new ArrayList<>();
            String folder = request.getFolderPath();

            // Carregar documentos de texto
            documents.addAll(FileSystemDocumentLoader.loadDocuments(folder, new TextDocumentParser()));

            // Carregar PDFs
            try {
                documents.addAll(FileSystemDocumentLoader.loadDocuments(folder, new ApachePdfBoxDocumentParser()));
            } catch (Exception e) {
                // Ignorar se não houver PDFs ou erro no parser
            }

            // Carregar Word/Office
            try {
                documents.addAll(FileSystemDocumentLoader.loadDocuments(folder, new ApachePoiDocumentParser()));
            } catch (Exception e) {
                // Ignorar
            }

            if (documents.isEmpty()) {
                return ResponseEntity.badRequest().body("Nenhum documento encontrado no caminho especificado.");
            }

            ragSystem.loadDocuments(documents);
            return ResponseEntity.ok("Successfully ingested " + documents.size() + " documents from " + folder);
        } catch (Exception e) {
            return ResponseEntity.internalServerError().body("Erro ao processar pasta: " + e.getMessage());
        }
    }

    @PostMapping("/query")
    public ResponseEntity<QueryResponse> query(@RequestBody QueryRequest request) {
        String answer = ragSystem.answer(request.getQuery());
        List<TextSegment> contexts = ragSystem.retrieveHybrid(request.getQuery(), 3, 0.5, 0.5);

        List<String> sources = contexts.stream()
                .map(TextSegment::text)
                .collect(Collectors.toList());

        return ResponseEntity.ok(new QueryResponse(answer, sources));
    }
}
