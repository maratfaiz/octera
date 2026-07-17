// Keep in sync with backend/app/services/segmentation.py's LAYER_LABELS_RU --
// the raw codes are an internal shorthand, not something to show a doctor.
// backend/tests/test_layer_labels_sync.py parses this literal and diffs it
// against the Python dict, so a drift on only one side fails CI.
export const LAYER_LABELS_RU: Record<string, string> = {
  nfl_gcl: "Слой нервных волокон / ганглиозных клеток (NFL/GCL)",
  ipl_inl: "Внутренний плексиформный / внутренний ядерный слой (IPL/INL)",
  opl_onl: "Наружный плексиформный / наружный ядерный слой (OPL/ONL)",
  photoreceptor: "Слой фоторецепторов",
  rpe: "Пигментный эпителий сетчатки (RPE)",
};
