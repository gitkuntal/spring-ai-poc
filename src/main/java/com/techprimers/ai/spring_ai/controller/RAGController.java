package com.techprimers.ai.spring_ai.controller;


import com.techprimers.ai.spring_ai.model.ChatResponse;
import com.techprimers.ai.spring_ai.model.QueryRequest;
import jakarta.annotation.PostConstruct;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.messages.Message;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.ai.chat.prompt.SystemPromptTemplate;
import org.springframework.ai.document.Document;
import org.springframework.ai.embedding.EmbeddingModel;
import org.springframework.ai.reader.TextReader;
import org.springframework.ai.transformer.splitter.TokenTextSplitter;
import org.springframework.ai.vectorstore.SimpleVectorStore;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.Resource;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

@RestController
public class RAGController {

    private final ChatClient chatClient;

    private final EmbeddingModel embeddingModel;

    private VectorStore vectorStore;

    private String template;

    @Value("${file.path}")
    private Resource resource;

    public RAGController(EmbeddingModel embeddingModel, ChatClient.Builder chatClientBuilder) {
        this.embeddingModel = embeddingModel;
        this.chatClient = chatClientBuilder.build();
    }

    @PostConstruct
    public void init() {
        vectorStore = new SimpleVectorStore(embeddingModel);
        TextReader textReader = new TextReader(resource);
        TokenTextSplitter textSplitter = new TokenTextSplitter();
        vectorStore.accept(textSplitter.apply(textReader.get()));
        template = """
                Answer the questions only using the information in the provided knowledge base.
                If you do not know the answer, please response with "I don't know."
                
                KNOWLEDGE BASE
                ---
                {documents}
                """;
    }

    @PostMapping("/rag")
    public ChatResponse rag(@RequestBody QueryRequest request) {
        // Retrieval
        String relevantDocs = vectorStore.similaritySearch(request.getQuery())
                .stream()
                .map(Document::getContent)
                .collect(Collectors.joining());

        // Augmented
        Message systemMessage = new SystemPromptTemplate(template)
                .createMessage(Map.of("documents", relevantDocs));

        // Generation
        Message userMessage = new UserMessage(request.getQuery());
        Prompt prompt = new Prompt(List.of(systemMessage, userMessage));
        ChatClient.ChatClientRequest.CallPromptResponseSpec res = chatClient.prompt(prompt).call();

        ChatResponse chatResponse = new ChatResponse();
        chatResponse.setResponseContent(res.content());

        return chatResponse;
    }
}
