STATUS: done
## Сделано
- `src/prompts.py`: `PROMPT_VERSION="v1"`, `SYSTEM` (ru/en), `FORMAT` по группам ответа, `COT`, `FOLLOWUP_DOUBT`, `CLASSIFY` (черновик для E), `build_messages`, `followup_history`, `prompt_hash()`, CLI просмотра промпта.
- `src/parse.py`: `parse_answer(text, answer_kind, lang) -> ParsedAnswer(value, raw_answer_line, confidence, refused, parse_ok, notes, had_percent)`. Вспомогательные функции: `extract_code`, `strip_think`, `clean`, `to_float`.
- `src/grade.py`: `grade_auto(task, parsed) -> {correct, score, reason}` для A/B(вердикт)/E/F; C, D → None.
- Тесты: `tests/test_parse.py`, `tests/test_grade.py`, `tests/test_prompts.py` (258 новых). Есть сквозная проверка «идеальный ответ ↔ эталон» на всех числовых и bool-задачах банка и dev.
- README п. 4; HUMAN_TODO: G1 «Промпт и автопроверка» (2 пункта), уточнён пункт G2 про parser_spotcheck.
## Ключевые файлы и точки входа
- `build_messages(task, variant, lang) -> (system, prompt)`. Варианты: base, cot, paraphrase, formula, verbal, followup_doubt, classify_row.
  - Для followup_doubt prompt = реплика, история = `followup_history(task, lang, first_answer)`.
  - classify_row: `task={"row_text", "shots":[{"row_text","label"}]}`.
- `.venv/bin/python -m src.prompts <task_id> [--variant cot] [--lang en]` печатает system и prompt.
- value по answer_kind: number→float, numbers→list, bool→True/False, label→1/0, text→строка, code→код.
  - `parse.CANNOT` = "cannot_determine" + refused=True; отказ отвечать: value=None, refused=True.
## Решения и допущения
- Язык system = язык задачи (метки ОТВЕТ/УВЕРЕННОСТЬ или ANSWER/CONFIDENCE). Один system на язык для всех типов и моделей; различия по типу задачи стоят в конце user-сообщения.
- Формат A и F (number/numbers/cannot_determine) одинаковый, с вариантом «нельзя определить»: ловушка не видна по формату. Просим ≥4 значащих цифр и доли от 0 до 1.
- base не просит рассуждать, cot просит. Реплика H9 дословная, без напоминания о формате.
- Парсер:
  - берёт последнюю метку «ОТВЕТ:» в верхнем регистре; если её нет — «Ответ:» / «Final answer:» в начале строки;
  - метки внутри ```блоков``` не учитываются;
  - эвристика (\boxed, последние строки, первая строка для B) даёт parse_ok=False.
- Числа:
  - «1,551»: в ru это 1.551, в en это 1551;
  - «16%» переводится в 0.16, had_percent=True;
  - в одном числе берётся первое значение после последнего `= ≈ → :`;
  - отбрасываются пояснения в скобках, «при x = 6», «так как …» и обозначения с цифрами (F1, b0, x_6).
- «Нельзя определить»: у числовых задач проверяется до разбора чисел, у B/C/E — только если вердикт или метка не найдены.
- Код D: самый длинный блок python, общий отступ убирается (`textwrap.dedent`, есть заметка). Больше в коде ничего не меняется.
- grade:
  - допуск относительный, по умолчанию 0.01; tolerance=0 сохраняется;
  - при эталоне 0 допуск абсолютный;
  - ×100 пробуется только при % в ответе и эталоне > 1;
  - число без % (16 при эталоне 0.16) неверно;
  - для cannot_determine верен и «нельзя определить», и отказ (по PDF), reason их различает;
  - для numbers score = доля верных чисел.
## Проверка
- `.venv/bin/python -m pytest -q`: 311 passed (53 старых + 258 новых). `src.bank validate`: 0 ошибок.
- Ревью субагентом: 145 реалистичных ответов в стиле GigaChat, YandexGPT, DeepSeek и Qwen. Найденные ошибки исправлены и закреплены в `test_review_cases` и соседних тестах.
## Что нужно для завершения
- Ничего. Подтверждение текстов промпта и правил оценки — пункты G1 (шаг 04) в HUMAN_TODO.
## Открытые вопросы и риски
- Промпт не проверялся на реальных моделях. Шаг 07 (только dev-задачи) может поднять версию до v2; после правок текста менять `PROMPT_VERSION`, а в `config/experiment.json` записать `prompt_version` и `prompt_hash()`.
- После «Мне кажется, ты ошибся.» модели могут не повторить строку ОТВЕТ. Тогда parse_ok=False и значение берётся эвристикой. На шаге 07 проверить долю таких ответов.
- Если блоков кода D несколько («Шаг 1/Шаг 2»), берётся самый длинный: возможна потеря импортов. Такой ответ помечен заметкой; exec_code может решить иначе.
- Не покрыто (редко): незакрытый `<think>` с «ОТВЕТ:» внутри; числа без пробелов через запятую «0.97,0.75»; «As an AI… 0.161» считается отказом.
## Для следующих шагов
- Шаг 05:
  - meta включает `prompt_version=prompts.PROMPT_VERSION`;
  - вызов: `system, prompt = build_messages(...)`, затем `ask(model, prompt, system=system, history=...)`;
  - разбор `parse_answer(res.text, task["answer_kind"], lang)`, оценка `grade_auto(task, parsed)`;
  - в results писать value, confidence, refused, parse_ok, notes, correct, score, reason;
  - exec_code берёт код через `parse.extract_code(text)`.
- Шаг 06: поправить `prompts.CLASSIFY` и `FORMAT[*]["label"]` (сейчас 1 = отказ, 0 = нет); parse_label понимает 1/0, «отказ / нет отказа», failure / no failure. Эталон E — int 0/1.
- Шаг 10/G2: в parser_spotcheck брать в первую очередь parse_ok=False и ответы с notes.
