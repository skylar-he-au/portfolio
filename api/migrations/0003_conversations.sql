-- One row per "Ask" conversation. The id is random and made by the server; it is also
-- LangGraph's thread_id, whose state (with the conversation's memory) the checkpointer
-- keeps in its own tables. Step 5c deletes conversations older than 24 hours, and their
-- checkpoints, by created_at; until then the API treats such an id as unknown.
CREATE TABLE conversations (
    id uuid PRIMARY KEY,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX conversations_created_at_idx ON conversations (created_at);
