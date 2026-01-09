package com.fatec.rag_hibrido.dto;

import java.util.List;
import java.util.Map;

public class IngestRequest {
    private List<DocumentDto> documents;

    public List<DocumentDto> getDocuments() {
        return documents;
    }

    public void setDocuments(List<DocumentDto> documents) {
        this.documents = documents;
    }

    public static class DocumentDto {
        private String content;
        private Map<String, String> metadata;

        public String getContent() {
            return content;
        }

        public void setContent(String content) {
            this.content = content;
        }

        public Map<String, String> getMetadata() {
            return metadata;
        }

        public void setMetadata(Map<String, String> metadata) {
            this.metadata = metadata;
        }
    }
}
