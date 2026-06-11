const LEGACY_CONFLICT_MAP: Record<string, string> = {
  temporal_scope: "freshness",
  false_premise: "misinformation",
  no_relevant_sources: "no_conflict",
};

const CONFLICT_LABELS: Record<string, string> = {
  no_conflict: "Không xung đột (No conflict)",
  misinformation: "Xung đột do thông tin sai lệch (Misinformation)",
  freshness: "Xung đột do thông tin lỗi thời (Outdated information / Freshness)",
  conflicting_opinions: "Xung đột quan điểm hoặc kết quả nghiên cứu (Conflicting opinions / Opinion)",
  complementary_information: "Thông tin bổ sung (Complementary Information)",
};

const CONFLICT_SHORT_LABELS: Record<string, string> = {
  no_conflict: "Không xung đột",
  misinformation: "Sai lệch",
  freshness: "Lỗi thời",
  conflicting_opinions: "Quan điểm",
  complementary_information: "Bổ sung",
};

const CONFLICT_SENTENCES: Record<string, string> = {
  no_conflict: "DRAG không phát hiện xung đột đáng kể giữa các nguồn, nên câu trả lời có thể tổng hợp trực tiếp từ bằng chứng đã truy hồi.",
  misinformation: "DRAG phát hiện khả năng có thông tin sai lệch hoặc tiền đề sai, nên câu trả lời cần sửa claim sai và dựa vào bằng chứng đáng tin cậy.",
  freshness: "DRAG phát hiện xung đột do thông tin lỗi thời hoặc khác phiên bản thời gian, nên câu trả lời cần ưu tiên nguồn mới hơn hoặc đúng mốc thời gian.",
  conflicting_opinions: "DRAG phát hiện các nguồn có quan điểm hoặc kết quả nghiên cứu trái chiều, nên câu trả lời cần trình bày trung lập các lập luận chính.",
  complementary_information: "DRAG phát hiện các nguồn bổ sung cho nhau, nên câu trả lời cần ghép các phần thông tin tương thích thành một kết luận thống nhất.",
};

export function normalizeConflictType(type?: string) {
  const mapped = LEGACY_CONFLICT_MAP[type || ""] || type || "no_conflict";
  return CONFLICT_LABELS[mapped] ? mapped : "no_conflict";
}

export function conflictLabel(type?: string) {
  return CONFLICT_LABELS[normalizeConflictType(type)];
}

export function conflictShortLabel(type?: string) {
  return CONFLICT_SHORT_LABELS[normalizeConflictType(type)];
}

export function conflictSentence(type?: string, fallback?: string) {
  const normalized = normalizeConflictType(type);
  if (type && normalizeConflictType(type) === type && fallback) return fallback;
  return CONFLICT_SENTENCES[normalized];
}
