export type SearchMode = "hybrid" | "keyword" | "vector";

export interface PaperSummary {
  id: number;
  arxiv_id: string;
  title: string;
  authors: string[];
  primary_category: string;
  categories: string[];
  published_at: string;
  abs_url: string;
  pdf_url: string | null;
  topic_id: number | null;
  topic_label: string | null;
}

export interface Topic {
  id: number;
  label: string;
  keywords: string[];
  size: number;
  x: number;
  y: number;
}

export interface PaperDetail extends PaperSummary {
  abstract: string;
  comment: string | null;
  doi: string | null;
  version: number;
  updated_at: string;
  chunks: { id: number; ordinal: number; text: string; n_words: number }[];
  topic: Topic | null;
  similar: PaperSummary[];
}

export interface SearchResult {
  paper: PaperSummary;
  score: number;
  snippet: string;
  chunk_id: number;
  keyword_rank: number | null;
  vector_rank: number | null;
  rerank_score: number | null;
}

export interface SearchResponse {
  query: string;
  mode: SearchMode;
  reranked: boolean;
  took_ms: number;
  timings: Record<string, number>;
  results: SearchResult[];
}

export interface Source {
  index: number;
  paper_id: number;
  arxiv_id: string;
  title: string;
  authors: string[];
  published_at: string;
  primary_category: string;
  abs_url: string;
  passages: string[];
  score: number;
}

export interface AskDone {
  provider: string;
  cited: number[];
  took_ms: number;
}

export interface TrendSeries {
  key: string;
  label: string;
  counts: number[];
  total: number;
  momentum: number;
}

export interface TrendsResponse {
  group_by: "topic" | "category";
  months: string[];
  totals: number[];
  series: TrendSeries[];
}

export interface TopicPoint {
  paper_id: number;
  arxiv_id: string;
  title: string;
  topic_id: number;
  x: number;
  y: number;
  published_at: string;
}

export interface Stats {
  papers: number;
  chunks: number;
  topics: number;
  categories: number;
  first_published: string | null;
  last_published: string | null;
}

export interface CategoryCount {
  category: string;
  count: number;
}

export interface Health {
  status: "ok" | "degraded";
  version: string;
  database: boolean;
  embedder: string | null;
  llm_provider: string | null;
  reranker: string | null;
}
