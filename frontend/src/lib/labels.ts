// Keep in sync with backend/app/services/segmentation.py's LAYER_LABELS_RU --
// the raw codes are an internal shorthand, not something to show a doctor.
// backend/tests/test_layer_labels_sync.py parses this literal and diffs it
// against the Python dict, so a drift on only one side fails CI.
export const LAYER_LABELS_RU: Record<string, string> = {
  nfl: "Слой нервных волокон (NFL)",
  gcl_ipl: "Слой ганглиозных клеток / внутренний плексиформный слой (GCL/IPL)",
  inl: "Внутренний ядерный слой (INL)",
  opl: "Наружный плексиформный слой (OPL)",
  onl: "Наружный ядерный слой (ONL)",
  is_os: "Слой фоторецепторов (IS/OS)",
  rpe: "Пигментный эпителий сетчатки (RPE)",
  choroid: "Хориоидея",
};
