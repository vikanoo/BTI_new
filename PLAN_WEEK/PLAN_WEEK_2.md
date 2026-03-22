# План разработки — Неделя 2 (23–29 марта 2026)

**Цель недели:** Полный сквозной поток — от сбора всех фото до итогового отчёта пользователю.
Подключить RAG, GPT-4o анализ и Python-микросервис для аннотации плана и PDF-конвертации.

**Предусловие:** К началу недели выполнены задачи Пн–Вс первой недели (Шаги 1–4, RAG-база нормативов).

---

## ✅ Понедельник 23.03 — RAG: подключение к N8N

**Supabase: функция поиска по векторам**

- [ ] Создать SQL-функцию `match_embeddings` в Supabase:
```sql
create or replace function match_embeddings(
  query_embedding vector(1536),
  match_count int default 5
)
returns table (id bigint, source text, chunk text, similarity float)
language sql stable as $$
  select id, source, chunk,
    1 - (embedding <=> query_embedding) as similarity
  from embeddings
  order by embedding <=> query_embedding
  limit match_count;
$$;
```

**N8N: узлы RAG-запроса**

- [x] Добавить узел OpenAI Embeddings — векторизовать контекстный запрос (описание изменений из промпта)
- [x] Добавить HTTP Request к Supabase RPC (`/rest/v1/rpc/match_embeddings`) с полученным вектором
- [ ] Проверить результат: должны возвращаться 3–5 релевантных фрагментов нормативов
- [x] Сформировать строку с нормами для вставки в промпт GPT-4o: `fragments.map(f => f.chunk).join('\n---\n')`

**Результат дня:** N8N умеет получать релевантные нормативы по заданному запросу

🖍️**Заметки:** Функцию `match_embeddings` необходимо создать вручную через Supabase Dashboard → SQL Editor (DDL недоступен через REST API). SQL приведён выше. Узлы RAG-запроса добавлены в воркфлоу: `Code - Prepare RAG Query` → `HTTP Request - Embed Query` → `HTTP Request - RAG Search` → `Code - Format RAG Results`. Подключены к `Telegram - All Photos Done`.

---

## ✅ Вторник 24.03 — Шаг 5: GPT-4o финальный анализ

**Ветка при `step = 'photos_complete'`**

- [x] Добавить `photos_complete` в `Switch1` (ветка photo/document) — переход к анализу
- [x] Добавить `photos_complete` в `Switch` (ветка text) — сообщение "анализ ещё выполняется"
- [x] Загрузить из `sessions`: `shots_json`, `plan_url`, `shots_status`
- [x] Сформировать список URL фотографий из `shots_status` (все со статусом `received`):
  `bti-files/photos/{chat_id}/{shot_id}.jpg`
- [x] Выполнить RAG-запрос (подготовить векторный запрос по контексту: "перепланировка квартиры, изменение стен, проёмов, санузла")
- [x] Вызвать GPT-4o с `detail:high`:
  - Системный промпт: роль эксперта по проверке перепланировок
  - Пользовательский промпт: план БТИ + все фотографии комнат + фрагменты нормативов
  - Запрос: вернуть JSON с полем `changes` (массив изменений с классификацией) и полем `confidence` (0–1)
- [x] Пример ожидаемого ответа GPT-4o:
```json
{
  "confidence": 0.85,
  "changes": [
    {
      "room_id": "room_1",
      "type": "wall_removed",
      "description": "Стена между жилой комнатой и кухней снесена",
      "classification": "illegal",
      "recommendation": "Требуется получение разрешения или демонтаж"
    }
  ],
  "summary": "Обнаружено 2 изменения, из которых 1 незаконное"
}
```
- [x] Сохранить результат в `sessions`: добавить поле `analysis_result` (JSONB) — HTTP PATCH к Supabase REST
- [x] Обновить `step = 'analysis_complete'`

**Результат дня:** GPT-4o анализирует план + фото + нормативы, возвращает структурированный JSON

---

## ✅ Среда 25.03 — Шаг 6: Отчёт пользователю + уведомление менеджера

**Формирование текстового отчёта**

- [x] Code Node: собрать текст отчёта из `analysis_result.changes`:
  - Заголовок с итоговым резюме
  - Нумерованный список изменений с иконками:
    - ✅ `согласуемая` — не требует действий
    - ⚠️ `требует_согласования` — нужно оформить документы
    - ❌ `незаконная` — затрагивает несущие конструкции / мокрые зоны
  - Для каждого пункта: описание + рекомендация
  - Дисклеймер: результат носит информационный характер, не является юридическим заключением
- [x] Отправить текстовый отчёт пользователю через Telegram

**Уведомление менеджера при низкой уверенности**

- [x] IF: `confidence < 0.6` → уведомить менеджера (Telegram сообщение на отдельный `chat_id` менеджера)
  - Сообщение: chat_id пользователя, краткое резюме, причина низкой уверенности
  - Переменная `MANAGER_CHAT_ID` — хранить в N8N Environment Variables
- [x] Пользователю при низкой уверенности: добавить пометку "Ваш запрос передан специалисту для ручной проверки"

**Сброс сессии**

- [x] Обновить `step = 'idle'`
- [x] Отправить финальное сообщение: "Анализ завершён. Для нового анализа отправьте /start"

**Результат дня:** Полный текстовый отчёт отправляется пользователю, менеджер уведомляется при необходимости

🖍️**Заметки:** Добавлено 8 узлов: `Code - Build Report` → `Telegram - Send Report` → `IF - Low Confidence` → ветка true: `Telegram - Notify Manager` → `HTTP Request - Reset Session (manager path)` → `Telegram - Analysis Done (manager path)`; ветка false: `HTTP Request - Reset Session (normal path)` → `Telegram - Analysis Done (normal path)`. Уведомление менеджера через `$env.MANAGER_CHAT_ID` (нужно задать в N8N Environment Variables). Сессия сбрасывается в `step = 'idle'`.

---

## ✅ Четверг 26.03 — Python-сервис v1: Docker + PDF-конвертация

**Инфраструктура Python-сервиса**

- [x] Создать Flask-приложение в Docker Compose (рядом с N8N):
  - `python-service/app.py` — основной файл
  - `python-service/requirements.txt` — Flask, pdf2image, Pillow
  - `python-service/Dockerfile` — python:3.11-slim + poppler-utils
  - `docker-compose.yml` — шаблон для добавления на VPS
- [x] Эндпоинт `POST /convert-pdf`:
  - Вход: PDF-файл (multipart/form-data, поле `file`)
  - Логика: `pdf2image.convert_from_bytes()` → первая страница → сохранить как PNG
  - Выход: PNG-файл (бинарный ответ, mimetype `image/png`)
- [ ] Протестировать эндпоинт вручную через curl/Postman (ручной шаг после деплоя на VPS)

**Интеграция PDF в бот**

- [x] Убрать заглушку "PDF не поддерживается" (узел `Send a text message3` отключён от цепочки)
- [x] Вместо этого: при `mime_type = application/pdf` → `Telegram - PDF Received` ("PDF получен, конвертирую...") → `Get Telegram File PDF` → `HTTP Request - Convert PDF` (POST к `http://python-service:5000/convert-pdf`) → `HTTP Request - Upload Converted Plan` (PUT в Supabase Storage как `plan.png`) → продолжить цепочку анализа
- [x] Добавить сообщение пользователю: "PDF получен, конвертирую в изображение..."

**Результат дня:** PDF-планы принимаются и конвертируются в PNG автоматически

🖍️**Заметки:** Добавлены 4 узла: `Telegram - PDF Received` → `Get Telegram File PDF` → `HTTP Request - Convert PDF` → `HTTP Request - Upload Converted Plan`. После загрузки PNG поток присоединяется к `HTTP Request - Download Plan` (та же точка, что и для обычных изображений). Тест через curl/Postman — ручной шаг после деплоя сервиса на VPS (`docker-compose up -d python-service`).

---

## ✅ Пятница 27.03 — Python-сервис v2: Аннотация плана

**Эндпоинт разметки комнат**

- [x] `POST /annotate-rooms`:
  - Вход: multipart/form-data `{ image: PNG, rooms_json: JSON string }`
  - Логика: PIL/Pillow — синие полупрозрачные прямоугольники по `region_percent` + подписи названий комнат
  - Выход: аннотированное PNG (бинарный ответ)
- [x] Интегрировать в N8N: после `Code - Parse Gemini` → `HTTP Request - Download Plan for Annotation` → `HTTP Request - Annotate Rooms` → `Telegram - Send Annotated Plan` (sendPhoto с подписью) → `Supabase - Save Analysis`

**Эндпоинт визуализации изменений**

- [x] `POST /annotate-changes`:
  - Вход: multipart/form-data `{ image: PNG, rooms_json: JSON string, changes: JSON string }`
  - Логика: по `room_id` находить `region_percent` из `rooms_json`, закрашивать:
    - Жёлтым (50% прозрачность) — `requires_approval`
    - Красным (50% прозрачность) — `illegal`
    - Зелёным (слабо) — `legal`
    - Нумерованные значки для каждого изменения
  - Выход: аннотированный план (PNG)
- [x] Интегрировать в N8N: после `Code - Build Report` → `HTTP Request - Download Plan for Changes` → `HTTP Request - Annotate Changes` → `Telegram - Send Annotated Changes` (sendPhoto) → `Telegram - Send Report` (текст) → `IF - Low Confidence`

**Результат дня:** Бот отправляет аннотированный план на шаге 2 и в финальном отчёте

🖍️**Заметки:** Добавлено 6 узлов в N8N. Шаг 2: план скачивается из Supabase Storage (public URL), передаётся в python-service, аннотированный PNG отправляется через `sendPhoto` вместо текстового списка. Шаг 6: аналогично — аннотированный план с изменениями отправляется перед текстовым отчётом. Заодно исправлен `.item` → `.first()` в `Code - Parse GPT4o Response`.

---

## Выходные 28–29.03 — Тестирование и доработки

**Сквозное тестирование**

- [ ] Провести полный тест с реальным планом БТИ (3-комнатная квартира):
  - `/start` → загрузить план → подождать Gemini → получить список комнат + аннотированный план
  - Отправить все фото по инструкциям → дождаться GPT-4o → получить отчёт + аннотированный план
- [ ] Проверить сценарий с PDF-планом
- [ ] Проверить уведомление менеджера при низкой уверенности

**Технические доработки**

- [ ] Исправить сохранение `plan_url` в сессии (сейчас `Update a row1` не сохраняет URL)
- [ ] Использовать полноразмерное фото (`photo[2]` или `photo[-1]`) вместо `photo[1]` при скачивании из Telegram
- [ ] Добавить обработку ошибок Gemini/GPT-4o: если API вернул ошибку — отправить пользователю сообщение и сбросить шаг
- [ ] Добавить тайм-аут сессии: если `step != 'idle'` и `updated_at` старше 48 часов — сбросить сессию
- [ ] Проверить и улучшить тексты сообщений бота (орфография, стиль, дисклеймеры)
- [ ] Добавить обработку случая, когда пользователь пишет что-то вне очереди при `step = 'collecting_photos'`

**Документация**

- [ ] Обновить `CLAUDE.md`: добавить Python-сервис в схему архитектуры, обновить статус
- [ ] Добавить инструкцию по добавлению `GEMINI_API_KEY` и `MANAGER_CHAT_ID` в N8N Environment Variables

---

## Итог недели

| День | Результат |
|------|-----------|
| ✅ Пн 23.03 | RAG-узлы добавлены в N8N (4 узла), match_embeddings SQL — ручной шаг в Supabase |
| ✅ Вт 24.03 | GPT-4o анализирует план + фото + RAG, возвращает JSON изменений |
| ✅ Ср 25.03 | Текстовый отчёт отправляется пользователю, менеджер уведомляется при низкой уверенности |
| ✅ Чт 26.03 | Python-сервис создан, PDF-интеграция в N8N готова (деплой на VPS — ручной шаг) |
| ✅ Пт 27.03 | Аннотированный план отправляется на шаге 2 и в финальном отчёте |
| Вых 28–29.03 | Полный сквозной тест пройден, критические баги исправлены |

**На следующей неделе (Phase 2):** PDF-отчёт, административный интерфейс обновления нормативов, многостраничные PDF-планы, региональные нормативы.

---

## Переменные окружения N8N (необходимо настроить до начала недели)

| Переменная | Описание |
|------------|----------|
| `GEMINI_API_KEY` | API-ключ Google Gemini |
| `OPENAI_API_KEY` | Уже должен быть в credentials N8N |
| `MANAGER_CHAT_ID` | Telegram chat_id менеджера для уведомлений |
| `PYTHON_SERVICE_URL` | URL Python-микросервиса (например, `http://python-service:5000`) |
