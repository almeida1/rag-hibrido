package com.fatec.rag_hibrido.model;

public class FolderIngestRequest {
    private String folderPath;
    private String globPattern; // Opcional, ex: "*.txt" ou "**/*.pdf"

    public String getFolderPath() {
        return folderPath;
    }

    public void setFolderPath(String folderPath) {
        this.folderPath = folderPath;
    }

    public String getGlobPattern() {
        return globPattern;
    }

    public void setGlobPattern(String globPattern) {
        this.globPattern = globPattern;
    }
}
