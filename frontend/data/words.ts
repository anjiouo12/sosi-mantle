import { Word } from "./types";

import { fireWords } from "./words-fire";
import { equipmentWords } from "./words-equipment";
import { rescueWords } from "./words-rescue";
import { disasterWords } from "./words-disaster";
import { facilityWords } from "./words-facility";
import { lawWords } from "./words-law";
import { buildingWords } from "./words-building";
import { generalWords } from "./words-general";


export const words: Word[] = [
  ...fireWords,
  ...equipmentWords,
  ...rescueWords,
  ...disasterWords,
  ...facilityWords,
  ...lawWords,
  ...buildingWords,
  ...generalWords,
];