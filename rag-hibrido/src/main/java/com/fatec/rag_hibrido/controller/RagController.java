package com.fatec.rag_hibrido.controller;

import com.fatec.rag_hibrido.model.FolderIngestRequest;
import com.fatec.rag_hibrido.model.IngestRequest;
import com.fatec.rag_hibrido.model.QueryRequest;
import com.fatec.rag_hibrido.model.QueryResponse;
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

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;

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

            Path folderPath = Paths.get(folder);

            if (!Files.exists(folderPath) || !Files.isDirectory(folderPath)) {
                return ResponseEntity.badRequest().body("Caminho inválido ou não é um diretório: " + folder);
            }

            try (Stream<Path> paths = Files.list(folderPath)) {
                paths.filter(Files::isRegularFile).forEach(path -> {
                    try {
                        String fileName = path.getFileName().toString().toLowerCase();
                        Document doc = null;
                        if (fileName.endsWith(".txt")) {
                            doc = FileSystemDocumentLoader.loadDocument(path, new TextDocumentParser());
                        } else if (fileName.endsWith(".pdf")) {
                            doc = FileSystemDocumentLoader.loadDocument(path, new ApachePdfBoxDocumentParser());
                        } else if (fileName.endsWith(".doc") || fileName.endsWith(".docx")) {
                            doc = FileSystemDocumentLoader.loadDocument(path, new ApachePoiDocumentParser());
                        }

                        if (doc != null) {
                            documents.add(doc);
                        }
                    } catch (Exception e) {
                        // Logar erro para arquivou específico mas continuar com os outros
                        System.err.println("Erro ao carregar documento " + path + ": " + e.getMessage());
                    }
                });
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
