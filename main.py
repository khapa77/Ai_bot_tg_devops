import logging
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, ReplyKeyboardMarkup, KeyboardButton
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
    CallbackQueryHandler,
    ConversationHandler
)
from openai import OpenAI
from datetime import datetime
import os
import asyncio
import json
import re
from contextlib import suppress

# Перед запуском установите переменные окружения BOT_TOKEN (токен Telegram) и OPENAI_API_KEY (ключ OpenAI)
"""Чтение секретов из файлов проекта и/или переменных окружения."""

def _read_secret_file(filename):
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(base_dir, filename)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    value = line.strip()
                    if value:
                        return value
    except Exception as e:
        logging.getLogger(__name__).warning(f"Не удалось прочитать {filename}: {e}")
    return None

# Настройки
_bot_token_from_file = _read_secret_file("tg_API")
BOT_TOKEN = _bot_token_from_file or os.environ.get("BOT_TOKEN")
if not BOT_TOKEN:
    raise RuntimeError("Токен Telegram не найден: задайте файл tg_API или переменную окружения BOT_TOKEN.")
ADMIN_IDS = [8345462682]  # Замените на ваш ID администратора
SYSTEM_PROMPT = """
Вы - полезный ассистент в Telegram боте. Отвечайте дружелюбно и информативно.
Избегайте вредных или опасных советов. Если вопрос непонятен, уточните.
Отвечайте на языке пользователя.
"""
CACHE_SIZE = 1000  # Количество кэшированных ответов
HISTORY_LENGTH = 5  # Количество сообщений для контекста

# Состояния для админ-панели
ADMIN_MENU, VIEW_STATS, BROADCAST = range(3)

# Настройка логгирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Инициализация AI клиентов с системным промтом
_openai_key_from_file = _read_secret_file("OpenAI_API")
OPENAI_API_KEY = _openai_key_from_file or os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Ключ OpenAI не найден: задайте файл OpenAI_API или переменную окружения OPENAI_API_KEY.")

# Дополнительно: поддержка project/organization для project-ключей (sk-proj-...)
_openai_project_from_file = _read_secret_file("OpenAI_PROJECT")
OPENAI_PROJECT = _openai_project_from_file or os.environ.get("OPENAI_PROJECT")

_openai_org_from_file = _read_secret_file("OpenAI_ORG")
OPENAI_ORG = _openai_org_from_file or os.environ.get("OPENAI_ORG")

ai_client = OpenAI(api_key=OPENAI_API_KEY, project=OPENAI_PROJECT, organization=OPENAI_ORG)  # ✅

# DeepSeek (опционально)
_deepseek_key_from_file = _read_secret_file("DeepSeek_API")
DEEPSEEK_API_KEY = _deepseek_key_from_file or os.environ.get("DEEPSEEK_API_KEY")
deepseek_client = None
if DEEPSEEK_API_KEY:
    try:
        deepseek_client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com/v1")
    except Exception as e:
        logger.warning(f"Не удалось инициализировать DeepSeek клиент: {e}")

default_system_prompt = """Ты — senior-админ и преподаватель с экспертизой в Linux, TCP/IP и Netflow. Твои ответы должны быть:

1. Ролевая модель:
- Говори как эксперт: "В ядре Linux это реализовано через..."
- Адаптируй уровень: новичкам — простое объяснение, экспертам — технические детали

2. Формат ответов:
1) Короткий ответ (1-2 предложения)
2) Подробное объяснение (при необходимости)
3) Готовые команды/конфиги
4) Практическое задание

3. Безопасность:
- Никогда не предлагать: rm -rf, chmod 777, iptables --flush
- Всегда объяснять риски и альтернативы

4. Работа с контекстом:
- Анализировать приложенные логи/конфиги построчно
- Для общих вопросов уточнять: "На каком дистрибутиве?", "Пришлите вывод `ip a`"
"""

# Загрузка списка промптов из файла promt_list (JSON)
PROMPTS = []  # список словарей: {id, title, content}
PROMPT_BY_ID = {}
USER_SELECTED_PROMPT = {}  # user_id -> prompt_id
USER_AI_PROVIDER = {}  # user_id -> 'OPEN_AI' | 'DEEP_SEEK'

# Совместимость для старых Python без asyncio.to_thread
try:
    _to_thread = asyncio.to_thread  # type: ignore[attr-defined]
except AttributeError:  # Python < 3.9
    from concurrent.futures import ThreadPoolExecutor
    _compat_executor = ThreadPoolExecutor(max_workers=4)
    async def _to_thread(func, *args, **kwargs):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(_compat_executor, lambda: func(*args, **kwargs))

TELEGRAM_MAX_MESSAGE_LEN = 4096
TELEGRAM_SAFE_SLICE_LEN = 3800

def _split_text_for_telegram(text: str, max_len: int = TELEGRAM_SAFE_SLICE_LEN) -> list:
    if not text:
        return [""]
    parts = []
    remaining = text
    while remaining:
        if len(remaining) <= max_len:
            parts.append(remaining)
            break
        cut = remaining.rfind("\n", 0, max_len)
        if cut == -1:
            cut = remaining.rfind(" ", 0, max_len)
            if cut == -1:
                cut = max_len
        parts.append(remaining[:cut])
        remaining = remaining[cut:].lstrip("\n ")
    return parts

def _is_region_block_error(err: Exception) -> bool:
    text = str(err).lower()
    return (
        "unsupported_country_region_territory" in text or
        "request_forbidden" in text or
        "403" in text and "openai" in text
    )

def _is_insufficient_balance_error(err: Exception) -> bool:
    text = str(err).lower()
    return ("402" in text) or ("insufficient" in text and "balance" in text) or ("payment required" in text)

def _fix_json_multiline_strings(text: str) -> str:
    """Грубая попытка исправить многострочные строки в JSON для ключей title/content.
    Заменяет реальные переводы строк внутри кавычек на символы \n.
    Это не полноценный парсер, но покрывает типовой случай.
    """
    def _replace_in_field(t: str, field: str) -> str:
        pattern = re.compile(rf'(\"{field}\"\s*:\s*\")(.*?)(\")', re.DOTALL)
        def _repl(m: re.Match) -> str:
            inner = m.group(2)
            inner_fixed = inner.replace("\n", r"\n")
            return f'{m.group(1)}{inner_fixed}{m.group(3)}'
        return pattern.sub(_repl, t)
    text = _replace_in_field(text, 'content')
    text = _replace_in_field(text, 'title')
    return text

def _load_prompts():
    global PROMPTS, PROMPT_BY_ID
    PROMPTS = []
    PROMPT_BY_ID = {}
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(base_dir, "promt_list")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                text = f.read().strip()
            if not text:
                raise ValueError("Файл promt_list пуст")
            try:
                data = json.loads(text)
            except Exception:
                # Попробуем поправить многострочные строки и распарсить снова
                fixed_text = _fix_json_multiline_strings(text)
                data = json.loads(fixed_text)
            # Поддерживаем два формата: список объектов или объект-словарь
            if isinstance(data, dict):
                # {"Title": "content", ...}
                for idx, (title, content) in enumerate(data.items()):
                    pid = f"p{idx+1}"
                    PROMPTS.append({"id": pid, "title": str(title), "content": str(content)})
            elif isinstance(data, list):
                # [{"title": ..., "content": ..., "id": optional}, ...]
                for idx, item in enumerate(data):
                    title = item.get("title") or f"Prompt {idx+1}"
                    content = item.get("content") or ""
                    pid = item.get("id") or f"p{idx+1}"
                    PROMPTS.append({"id": str(pid), "title": str(title), "content": str(content)})
            else:
                raise ValueError("Неподдерживаемый формат promt_list")
        else:
            # Файл отсутствует — используем дефолтный промпт
            PROMPTS = [{"id": "default", "title": "Стандартный промпт", "content": default_system_prompt}]
        # Построить индекс
        PROMPT_BY_ID = {p["id"]: p for p in PROMPTS}
        if not PROMPTS:
            PROMPTS = [{"id": "default", "title": "Стандартный промпт", "content": default_system_prompt}]
            PROMPT_BY_ID = {"default": PROMPTS[0]}
        logger.info(f"Загружено промптов: {len(PROMPTS)}")
    except Exception as e:
        logger.warning(f"Ошибка загрузки promt_list: {e}. Использую дефолтный промпт.")
        PROMPTS = [{"id": "default", "title": "Стандартный промпт", "content": default_system_prompt}]
        PROMPT_BY_ID = {"default": PROMPTS[0]}

def _get_user_system_prompt(user_id: int) -> str:
    pid = USER_SELECTED_PROMPT.get(user_id)
    if pid and pid in PROMPT_BY_ID:
        return PROMPT_BY_ID[pid]["content"]
    return default_system_prompt

def _get_user_ai_provider(user_id: int) -> str:
    provider = USER_AI_PROVIDER.get(user_id)
    if provider in ("OPEN_AI", "DEEP_SEEK"):
        return provider
    return "OPEN_AI"

def _get_client_and_model(user_id: int, vision: bool = False):
    provider = _get_user_ai_provider(user_id)
    if provider == "DEEP_SEEK":
        if not deepseek_client:
            raise RuntimeError("DeepSeek не настроен: задайте ключ в файле DeepSeek_API или переменной DEEPSEEK_API_KEY")
        model = "deepseek-chat"  # базовая модель чата DeepSeek
        if vision:
            # На текущий момент Vision может быть недоступен у DeepSeek
            return deepseek_client, model, False
        return deepseek_client, model, True
    # OPEN_AI по умолчанию
    if vision:
        return ai_client, "gpt-4-vision-preview", True
    return ai_client, "gpt-4-turbo", True

# PDF отчёты (опционально, через reportlab)
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False

def _ensure_reports_dir() -> str:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base_dir, "temp_reports")
    os.makedirs(path, exist_ok=True)
    return path

# Регистрация шрифта с поддержкой кириллицы
CYR_FONT = "Helvetica"
CYR_FONT_BOLD = "Helvetica-Bold"

def _register_cyrillic_fonts():
    global CYR_FONT, CYR_FONT_BOLD
    if not REPORTLAB_AVAILABLE:
        return
    candidates = [
        ("DejaVuSans", [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/local/share/fonts/DejaVuSans.ttf",
        ]),
        ("NotoSans", [
            "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
            "/usr/local/share/fonts/NotoSans-Regular.ttf",
        ]),
        ("FreeSans", [
            "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
            "/usr/local/share/fonts/FreeSans.ttf",
        ]),
    ]
    bold_candidates = [
        ("DejaVuSans-Bold", [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/local/share/fonts/DejaVuSans-Bold.ttf",
        ]),
        ("NotoSans-Bold", [
            "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
            "/usr/local/share/fonts/NotoSans-Bold.ttf",
        ]),
        ("FreeSansBold", [
            "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
            "/usr/local/share/fonts/FreeSansBold.ttf",
        ]),
    ]
    def try_register(name: str, paths: list) -> bool:
        for p in paths:
            if os.path.exists(p):
                try:
                    pdfmetrics.registerFont(TTFont(name, p))
                    return True
                except Exception:
                    continue
        return False
    # Обычный
    for name, paths in candidates:
        if try_register(name, paths):
            CYR_FONT = name
            break
    # Жирный
    for name, paths in bold_candidates:
        if try_register(name, paths):
            CYR_FONT_BOLD = name
            break

def _generate_user_report_pdf(user_id: int) -> str:
    if not REPORTLAB_AVAILABLE:
        raise RuntimeError("reportlab не установлен. Установите: pip install reportlab")
    reports_dir = _ensure_reports_dir()
    _register_cyrillic_fonts()
    file_path = os.path.join(reports_dir, f"report_user_{user_id}.pdf")
    c = canvas.Canvas(file_path, pagesize=A4)
    width, height = A4
    c.setTitle("User Report")
    c.setFont(CYR_FONT_BOLD, 16)
    c.drawString(40, height - 50, "Индивидуальный отчет пользователя")
    c.setFont(CYR_FONT, 12)
    c.drawString(40, height - 90, f"User ID: {user_id}")
    # Немного данных статистики
    total_msgs = bot_stats.get("total_messages", 0)
    active_users = len(bot_stats.get("active_users", []))
    c.drawString(40, height - 120, f"Всего сообщений в боте: {total_msgs}")
    c.drawString(40, height - 140, f"Уникальных пользователей: {active_users}")
    # Последние сообщения пользователя
    c.drawString(40, height - 180, "Последние сообщения (до 5):")
    y = height - 200
    ctx = user_contexts.get(user_id, [])[-5:]
    for msg in ctx:
        line = f"{msg.get('role')}: {msg.get('content')[:90]}"  # обрезаем для простоты
        c.drawString(50, y, line)
        y -= 20
        if y < 60:
            c.showPage()
            c.setFont(CYR_FONT, 12)
            y = height - 60
    c.showPage()
    c.save()
    return file_path

def _generate_admin_report_pdf() -> str:
    if not REPORTLAB_AVAILABLE:
        raise RuntimeError("reportlab не установлен. Установите: pip install reportlab")
    reports_dir = _ensure_reports_dir()
    _register_cyrillic_fonts()
    file_path = os.path.join(reports_dir, "report_admin.pdf")
    c = canvas.Canvas(file_path, pagesize=A4)
    width, height = A4
    c.setTitle("Admin Report")
    c.setFont(CYR_FONT_BOLD, 16)
    c.drawString(40, height - 50, "Сводный отчет по боту")
    c.setFont(CYR_FONT, 12)
    c.drawString(40, height - 90, f"Всего сообщений: {bot_stats.get('total_messages', 0)}")
    c.drawString(40, height - 110, f"Уникальных пользователей: {len(bot_stats.get('active_users', []))}")
    last_active = max(bot_stats.get('last_active', {}).values(), default='нет данных')
    c.drawString(40, height - 130, f"Последняя активность: {last_active}")
    c.showPage()
    c.save()
    return file_path

async def get_ai_response(messages: list) -> str:
    # Выбор клиента/модели на основе первого сообщения system, далее по user_id будет корректнее
    # Здесь уточнение идёт в вызывающих местах — мы туда передадим user_id для маршрутизации.
    raise RuntimeError("get_ai_response используется через get_ai_response_for_user")
    return response.choices[0].message.content




# А системный промт передавайте в messages через _get_user_system_prompt


# Кэш для ответов (асинхронный, по tuple сообщений и провайдеру)
ai_response_cache = {}

async def get_cached_ai_response_for_user(user_id: int, messages: list) -> str:
    client, model, supported = _get_client_and_model(user_id, vision=False)
    # Ключ — провайдер+модель + tuple из (role, content) для каждого сообщения
    cache_key = (model, tuple((msg['role'], msg['content']) for msg in messages))
    if cache_key in ai_response_cache:
        return ai_response_cache[cache_key]
    if not supported:
        raise RuntimeError("Выбранный провайдер не поддерживает чат-модели")
    response_obj = await _to_thread(
        client.chat.completions.create,
        model=model,
        messages=messages,
        temperature=0.7
    )
    response = response_obj.choices[0].message.content
    ai_response_cache[cache_key] = response
    return response

# Хранение контекста диалогов
user_contexts = {}

# Статистика бота
bot_stats = {
    "total_messages": 0,
    "active_users": set(),
    "last_active": {}
}

def update_stats(user_id: int):
    """Обновляем статистику"""
    bot_stats["total_messages"] += 1
    bot_stats["active_users"].add(user_id)
    bot_stats["last_active"][user_id] = datetime.now().isoformat()

# Обработчик команды /start
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        user = update.effective_user
        welcome_message = (
            f"Привет, {user.first_name}! 👋\n\n"
            "Я умный бот с искусственным интеллектом. Можете:\n"
            "- Задавать вопросы\n"
            "- Отправлять изображения для анализа\n"
            "- Вести диалог с контекстом\n\n"
            "Просто напишите или отправьте что-нибудь!"
        )
        reply_kb = ReplyKeyboardMarkup(
            [
                [KeyboardButton("💬 Задать вопрос"), KeyboardButton("🧠 Выбрать промпт")],
                [KeyboardButton("📊 Статистика"), KeyboardButton("🧹 Сброс контекста")],
                [KeyboardButton("🤖 Выбрать AI"), KeyboardButton("📄 Мой отчет")]
            ],
            resize_keyboard=True
        )
        await update.message.reply_text(welcome_message, reply_markup=reply_kb)
        update_stats(user.id)
        logger.info(f"/start от пользователя {user.id}")
    except Exception as e:
        logger.error(f"Ошибка в start: {e}")
        if update and hasattr(update, 'message'):
            await update.message.reply_text("Ошибка при запуске. Попробуйте позже.")

async def menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    reply_kb = ReplyKeyboardMarkup(
        [
            [KeyboardButton("💬 Задать вопрос"), KeyboardButton("🧠 Выбрать промпт")],
            [KeyboardButton("📊 Статистика"), KeyboardButton("🧹 Сброс контекста")],
            [KeyboardButton("🤖 Выбрать AI"), KeyboardButton("📄 Мой отчет")]
        ],
        resize_keyboard=True
    )
    await update.message.reply_text("Главное меню:", reply_markup=reply_kb)

async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Доступные команды:\n"
        "/start — приветствие и меню\n"
        "/menu — показать меню\n"
        "/prompt — выбрать системный промпт\n"
        "/ai — выбрать AI провайдера (OpenAI / DeepSeek)\n"
        "/myreport — PDF отчет для пользователя\n"
        "/report — PDF отчет для админа\n"
        "/reload_prompts — перезагрузить список промптов (админ)\n"
        "/admin — админ-панель\n"
        "/reset — очистить контекст диалога"
    )

async def reset_context(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_contexts[user_id] = []
    await update.message.reply_text("Контекст диалога очищен.")

async def my_report(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    try:
        if not REPORTLAB_AVAILABLE:
            await update.message.reply_text("reportlab не установлен. Установите: pip install reportlab")
            return
        file_path = _generate_user_report_pdf(user_id)
        with open(file_path, "rb") as f:
            await update.message.reply_document(document=f, filename=os.path.basename(file_path))
    except Exception as e:
        logger.error(f"Ошибка генерации пользовательского отчёта: {e}")
        await update.message.reply_text("Не удалось создать PDF отчет.")

async def admin_report(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id not in ADMIN_IDS:
        await update.message.reply_text("Доступ запрещен.")
        return
    try:
        if not REPORTLAB_AVAILABLE:
            await update.message.reply_text("reportlab не установлен. Установите: pip install reportlab")
            return
        file_path = _generate_admin_report_pdf()
        with open(file_path, "rb") as f:
            await update.message.reply_document(document=f, filename=os.path.basename(file_path))
    except Exception as e:
        logger.error(f"Ошибка генерации админского отчёта: {e}")
        await update.message.reply_text("Не удалось создать PDF отчет.")

async def handle_menu_buttons(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (update.message.text or "").strip()
    if text == "🧠 Выбрать промпт":
        return await prompt_menu(update, context)
    if text == "🤖 Выбрать AI":
        return await ai_menu(update, context)
    if text == "📊 Статистика":
        # Покажем краткую статистику
        stats_text = (
            f"📊 Статистика бота:\n"
            f"• Всего сообщений: {bot_stats['total_messages']}\n"
            f"• Уникальных пользователей: {len(bot_stats['active_users'])}"
        )
        return await update.message.reply_text(stats_text)
    if text == "🧹 Сброс контекста":
        return await reset_context(update, context)
    if text == "📄 Мой отчет":
        return await my_report(update, context)
    if text == "💬 Задать вопрос":
        return await update.message.reply_text("Напишите свой вопрос сообщением ниже.")
    # иначе — пропускаем дальше к обычной обработке текста
    return await handle_text(update, context)

# Обработчик текстовых сообщений с контекстом
async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    user_id = user.id
    user_message = update.message.text
    
    update_stats(user_id)
    
    # Получаем или создаем контекст пользователя
    if user_id not in user_contexts:
        user_contexts[user_id] = []
    
    # Добавляем новое сообщение в контекст (ограничиваем длину)
    user_contexts[user_id].append({"role": "user", "content": user_message})
    user_contexts[user_id] = user_contexts[user_id][-HISTORY_LENGTH:]
    
    # Формируем messages с учётом выбранного промпта и провайдера
    system_prompt_text = _get_user_system_prompt(user_id)
    messages = [{"role": "system", "content": system_prompt_text}] + user_contexts[user_id]
    
    try:
        logger.info(f"Processing message from {user_id}: {user_message}")
        
        # Получаем ответ (из кэша или API), учитывая выбранного провайдера
        ai_response = await get_cached_ai_response_for_user(user_id, messages)
        
        # Добавляем ответ в контекст
        user_contexts[user_id].append({"role": "assistant", "content": ai_response})
        
        # Кнопка для сохранения ответа в PDF
        reply_markup = InlineKeyboardMarkup(
            [[InlineKeyboardButton("Сохранить ответ в PDF", callback_data="save_pdf")]]
        )
        # Отправляем ответ частями, чтобы не превысить ограничения Telegram
        chunks = _split_text_for_telegram(ai_response)
        if chunks:
            # Первая часть с кнопкой PDF
            await update.message.reply_text(chunks[0], reply_markup=reply_markup)
            # Остальные части без кнопки
            for chunk in chunks[1:]:
                await update.message.reply_text(chunk)
        
    except Exception as e:
        if _is_insufficient_balance_error(e):
            logger.error(f"Provider insufficient balance: {e}")
            await update.message.reply_text(
                "У провайдера недостаточно средств/кредита. Пополните баланс или выберите другого провайдера через /ai."
            )
        elif _is_region_block_error(e):
            logger.error(f"OpenAI region restriction: {e}")
            await update.message.reply_text(
                "Доступ к OpenAI ограничен в вашем регионе. Варианты:\n"
                "- Запустите бота на хосте в поддерживаемом регионе\n"
                "- Используйте Azure OpenAI (потребуется другой клиент/ключ)"
            )
        else:
            logger.error(f"Error processing message: {e}")
            await update.message.reply_text("Извините, произошла ошибка. Попробуйте позже.")

async def analyze_image_with_openai(image_path: str) -> str:
    # Используем OpenAI Vision (gpt-4-vision-preview)
    with open(image_path, "rb") as img_file:
        # Для простоты используем текущий промпт без привязки к пользователю в этом helper.
        # Выбор промпта учитывается в handle_image ниже посредством user_id.
        content_bytes = img_file.read()
        response = await _to_thread(
            ai_client.chat.completions.create,
            model="gpt-4-vision-preview",
            messages=[
                {"role": "system", "content": default_system_prompt},
                {"role": "user", "content": "Что на этом изображении?", "image": content_bytes}
            ],
            max_tokens=512
        )
    return response.choices[0].message.content

# Обработчик изображений
async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    if not update.message.photo:
        await update.message.reply_text("Фото не найдено.")
        return
    photo = update.message.photo[-1]  # Берем самое высокое качество
    try:
        # Убедимся, что директория temp_images существует
        os.makedirs("temp_images", exist_ok=True)
        # Скачиваем изображение
        photo_file = await photo.get_file()
        image_path = f"temp_images/{user.id}_{photo.file_id}.jpg"
        await photo_file.download_to_drive(image_path)
        # Отправляем изображение в AI API с учётом выбранного промпта пользователя
        # Здесь используем выбранный промпт как системный
        user_prompt = _get_user_system_prompt(user.id)
        with open(image_path, "rb") as img_file:
            content_bytes = img_file.read()
        client, model, supported = _get_client_and_model(user.id, vision=True)
        if not supported:
            raise RuntimeError("Выбранный провайдер не поддерживает анализ изображений")
        response_obj = await _to_thread(
            client.chat.completions.create,
            model=model,
            messages=[
                {"role": "system", "content": user_prompt},
                {"role": "user", "content": "Что на этом изображении?", "image": content_bytes}
            ],
            max_tokens=512
        )
        response = response_obj.choices[0].message.content
        # Удаляем временный файл
        os.remove(image_path)
        await update.message.reply_text(response)
    except Exception as e:
        if _is_insufficient_balance_error(e):
            logger.error(f"Provider insufficient balance: {e}")
            await update.message.reply_text(
                "У провайдера недостаточно средств/кредита. Пополните баланс или выберите другого провайдера через /ai."
            )
        elif _is_region_block_error(e):
            logger.error(f"OpenAI region restriction: {e}")
            await update.message.reply_text(
                "Доступ к OpenAI ограничен в вашем регионе. Перенесите запуск бота в поддерживаемый регион или используйте Azure OpenAI."
            )
        else:
            logger.error(f"Error processing image: {e}")
            await update.message.reply_text("Не удалось обработать изображение.")

# Админ-панель
async def admin_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id not in ADMIN_IDS:
        await update.message.reply_text("Доступ запрещен.")
        return ConversationHandler.END
    
    keyboard = [
        [InlineKeyboardButton("Статистика", callback_data="view_stats")],
        [InlineKeyboardButton("Рассылка", callback_data="broadcast")],
        [InlineKeyboardButton("Выход", callback_data="cancel")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.message.reply_text(
        "Админ-панель:",
        reply_markup=reply_markup
    )
    
    return ADMIN_MENU

async def view_stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    
    stats_text = (
        f"📊 Статистика бота:\n"
        f"• Всего сообщений: {bot_stats['total_messages']}\n"
        f"• Уникальных пользователей: {len(bot_stats['active_users'])}\n"
        f"• Последняя активность: {max(bot_stats['last_active'].values(), default='нет данных')}"
    )
    
    await query.edit_message_text(stats_text)
    return ADMIN_MENU

# Выбор промпта — команда и обработчики
async def prompt_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    try:
        # Собираем клавиатуру из списка промптов
        keyboard = []
        for p in PROMPTS:
            title = p["title"]
            pid = p["id"]
            prefix = "✅ " if USER_SELECTED_PROMPT.get(user_id) == pid else ""
            keyboard.append([InlineKeyboardButton(f"{prefix}{title}", callback_data=f"set_prompt:{pid}")])
        reply_markup = InlineKeyboardMarkup(keyboard)
        await update.message.reply_text("Выберите промпт:", reply_markup=reply_markup)
    except Exception as e:
        logger.error(f"Ошибка формирования меню промптов: {e}")
        await update.message.reply_text("Не удалось загрузить список промптов.")

async def ai_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    try:
        keyboard = []
        current = _get_user_ai_provider(user_id)
        # OpenAI всегда доступен
        prefix_oa = "✅ " if current == "OPEN_AI" else ""
        keyboard.append([InlineKeyboardButton(f"{prefix_oa}OpenAI", callback_data="set_ai:OPEN_AI")])
        # DeepSeek — только если настроен ключ
        if deepseek_client is not None:
            prefix_ds = "✅ " if current == "DEEP_SEEK" else ""
            keyboard.append([InlineKeyboardButton(f"{prefix_ds}DeepSeek", callback_data="set_ai:DEEP_SEEK")])
        reply_markup = InlineKeyboardMarkup(keyboard)
        await update.message.reply_text("Выберите AI провайдера:", reply_markup=reply_markup)
    except Exception as e:
        logger.error(f"Ошибка формирования меню AI: {e}")
        await update.message.reply_text("Не удалось загрузить список провайдеров.")

async def set_prompt_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    try:
        data = query.data
        if not data.startswith("set_prompt:"):
            return
        pid = data.split(":", 1)[1]
        user_id = query.from_user.id
        if pid in PROMPT_BY_ID:
            USER_SELECTED_PROMPT[user_id] = pid
            title = PROMPT_BY_ID[pid]["title"]
            await query.edit_message_text(f"Выбран промпт: {title}")
        else:
            await query.edit_message_text("Неизвестный промпт.")
    except Exception as e:
        logger.error(f"Ошибка выбора промпта: {e}")
        await query.edit_message_text("Не удалось применить промпт.")

async def set_ai_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    try:
        data = query.data
        if not data.startswith("set_ai:"):
            return
        provider = data.split(":", 1)[1]
        user_id = query.from_user.id
        if provider not in ("OPEN_AI", "DEEP_SEEK"):
            await query.edit_message_text("Неизвестный провайдер.")
            return
        if provider == "DEEP_SEEK" and deepseek_client is None:
            await query.edit_message_text("DeepSeek не настроен. Добавьте ключ DEEPSEEK_API_KEY.")
            return
        USER_AI_PROVIDER[user_id] = provider
        await query.edit_message_text(f"Выбран AI провайдер: {'OpenAI' if provider=='OPEN_AI' else 'DeepSeek'}")
    except Exception as e:
        logger.error(f"Ошибка выбора AI: {e}")
        await query.edit_message_text("Не удалось применить провайдера.")

async def save_pdf_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    user_id = query.from_user.id
    try:
        if not REPORTLAB_AVAILABLE:
            await query.edit_message_text("reportlab не установлен. Установите: pip install reportlab")
            return
        # Берём последний ответ ассистента из контекста
        ctx = user_contexts.get(user_id, [])
        last_answer = next((m["content"] for m in reversed(ctx) if m.get("role") == "assistant"), None)
        if not last_answer:
            await query.edit_message_text("Нет ответа для сохранения.")
            return
        # Генерируем одноразовый PDF с текстом ответа
        reports_dir = _ensure_reports_dir()
        _register_cyrillic_fonts()
        file_path = os.path.join(reports_dir, f"answer_{user_id}.pdf")
        c = canvas.Canvas(file_path, pagesize=A4)
        width, height = A4
        c.setTitle("AI Answer")
        c.setFont(CYR_FONT_BOLD, 14)
        c.drawString(40, height - 50, "Ответ ассистента")
        c.setFont(CYR_FONT, 12)
        # Простейшая разбивка текста по строкам
        margin_left = 40
        y = height - 80
        max_width = width - 80
        for paragraph in last_answer.split("\n"):
            line = ""
            for word in paragraph.split(" "):
                test = (line + (" " if line else "") + word).strip()
                if c.stringWidth(test, CYR_FONT, 12) <= max_width:
                    line = test
                else:
                    c.drawString(margin_left, y, line)
                    y -= 18
                    if y < 60:
                        c.showPage()
                        c.setFont(CYR_FONT, 12)
                        y = height - 60
                    line = word
            if line:
                c.drawString(margin_left, y, line)
                y -= 18
                if y < 60:
                    c.showPage()
                    c.setFont(CYR_FONT, 12)
                    y = height - 60
        c.showPage()
        c.save()
        # Отправляем файл
        with open(file_path, "rb") as f:
            await query.message.reply_document(document=f, filename=os.path.basename(file_path))
        await query.edit_message_text("PDF сформирован и отправлен.")
    except Exception as e:
        logger.error(f"Ошибка создания PDF ответа: {e}")
        await query.edit_message_text("Не удалось сформировать PDF.")

async def reload_prompts(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id not in ADMIN_IDS:
        await update.message.reply_text("Доступ запрещен.")
        return
    _load_prompts()
    await update.message.reply_text(f"Промпты перезагружены. Доступно: {len(PROMPTS)}")

async def start_broadcast(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    
    await query.edit_message_text("Введите сообщение для рассылки:")
    return BROADCAST

async def process_broadcast(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.message.text
    users = bot_stats["active_users"]
    async def send_msg(user_id):
        try:
            await context.bot.send_message(user_id, f"📢 Рассылка:\n\n{message}")
        except Exception as e:
            logger.error(f"Failed to send to {user_id}: {e}")
    await asyncio.gather(*(send_msg(uid) for uid in users))
    await update.message.reply_text(f"Рассылка отправлена {len(users)} пользователям.")
    return ConversationHandler.END

async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    
    await query.edit_message_text("Админ-панель закрыта.")
    return ConversationHandler.END

# Обработчик ошибок
async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    logger.error(f"Update {update} caused error {context.error}")
    if update and hasattr(update, 'message'):
        await update.message.reply_text("Произошла ошибка. Пожалуйста, попробуйте еще раз.")

def main():
    # Создаем папку для временных изображений (на всякий случай)
    os.makedirs("temp_images", exist_ok=True)
    
    # Инициализируем список промптов
    _load_prompts()
    
    # Создаем Application
    application = Application.builder().token(BOT_TOKEN).build()
    
    # Обработчики команд
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("menu", menu))
    application.add_handler(CommandHandler("help", help_cmd))
    application.add_handler(CommandHandler("reset", reset_context))
    application.add_handler(CommandHandler("myreport", my_report))
    application.add_handler(CommandHandler("report", admin_report))
    application.add_handler(CommandHandler("admin", admin_menu))
    application.add_handler(CommandHandler("prompt", prompt_menu))
    application.add_handler(CommandHandler("reload_prompts", reload_prompts))
    
    # Обработчики сообщений
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_menu_buttons))
    application.add_handler(MessageHandler(filters.PHOTO, handle_image))
    
    # Обработчик админ-панели
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("admin", admin_menu)],
        states={
            ADMIN_MENU: [
                CallbackQueryHandler(view_stats, pattern="^view_stats$"),
                CallbackQueryHandler(start_broadcast, pattern="^broadcast$"),
                CallbackQueryHandler(cancel, pattern="^cancel$")
            ],
            BROADCAST: [MessageHandler(filters.TEXT & ~filters.COMMAND, process_broadcast)]
        },
        fallbacks=[CommandHandler("cancel", cancel)]
    )
    application.add_handler(conv_handler)
    application.add_handler(CallbackQueryHandler(set_prompt_callback, pattern=r"^set_prompt:"))
    application.add_handler(CallbackQueryHandler(save_pdf_callback, pattern=r"^save_pdf$"))
    application.add_handler(CallbackQueryHandler(set_ai_callback, pattern=r"^set_ai:"))
    
    # Обработчик ошибок
    application.add_error_handler(error_handler)
    
    # Запускаем бота
    logger.info("Bot is starting...")
    application.run_polling()

if __name__ == '__main__':
    main()
