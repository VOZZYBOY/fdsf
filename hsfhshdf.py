import aiohttp
import asyncio
import aiofiles
import json
import uvicorn
import logging
import time
import os
import numpy as np
import re
import pickle
from pathlib import Path
from fastapi import FastAPI, HTTPException, Form, UploadFile, File
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from voicerecognise import recognize_audio_with_sdk
from yandex_cloud_ml_sdk import YCloudML
from typing import Dict, List, Optional
import faiss  

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

BASE_DIR = "base"
EMBEDDINGS_DIR = "embeddings_data"
os.makedirs(BASE_DIR, exist_ok=True)
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
API_URL = ""
YANDEX_FOLDER_ID = ""
YANDEX_API_KEY = ""
API_BASE_URL = "https://dev.back.matrixcrm.ru/api/v1/AI"  # Базовый URL для API
DEFAULT_TENANT_ID = "medyumed.2023-04-24"  # Дефолтный tenant_id
DEFAULT_LANG_ID = "ru"  # Дефолтный язык
DEFAULT_COLOR_CODE = "grey"  # Дефолтный цвет для записи
DEFAULT_TRAFFIC_CHANNEL = 0  # Дефолтный канал трафика
logger.info("Загрузка моделей...")
search_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
logger.info("Модели успешно загружены.")
conversation_history: Dict[str, Dict] = {}
app = FastAPI()


def get_tenant_path(tenant_id: str) -> Path:
    """Создает папку для конкретного тенанта"""
    tenant_path = Path(EMBEDDINGS_DIR) / tenant_id
    tenant_path.mkdir(parents=True, exist_ok=True)
    return tenant_path


def normalize_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"[^\w\s\d\n]", "", text)
    return text.lower()


def tokenize_text(text: str) -> List[str]:
    stopwords = {
        "и", "в", "на", "с", "по", "для", "как", "что", "это", "но",
        "а", "или", "у", "о", "же", "за", "к", "из", "от", "так", "то", "все"
    }
    tokens = text.split()
    return [word for word in tokens if word not in stopwords]


def extract_text_fields(record: dict) -> str:
    """
    Формирует многострочное текстовое представление записи, сохраняющее иерархию:
      Филиал: <filialName>
      Категория: <categoryName>
      Услуга: <serviceName>
      Описание услуги: <serviceDescription>
      Цена: <price>
      Специалист: <employeeFullName>
      Описание специалиста: <employeeDescription>
    """
    filial = record.get("filialName", "Филиал не указан")
    category = record.get("categoryName", "Категория не указана")
    service = record.get("serviceName", "Услуга не указана")
    service_desc = record.get("serviceDescription", "Описание услуги не указано")
    price = record.get("price", "Цена не указана")
    specialist = record.get("employeeFullName", "Специалист не указан")
    spec_desc = record.get("employeeDescription", "Описание не указано")
    text = (
        f"Филиал: {filial}\n"
        f"Категория: {category}\n"
        f"Услуга: {service}\n"
        f"Описание услуги: {service_desc}\n"
        f"Цена: {price}\n"
        f"Специалист: {specialist}\n"
        f"Описание специалиста: {spec_desc}"
    )
    return normalize_text(text)


async def load_json_data(tenant_id: str) -> List[dict]:
    """
    Загружает данные из JSON-файла и преобразует их в список записей.
    """
    file_path = os.path.join(BASE_DIR, f"{tenant_id}.json")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"Файл для tenant_id={tenant_id} не найден.")
    
    async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
        content = await f.read()
        data = json.loads(content)
    
    records = []
    branches = data.get("data", {}).get("branches", [])
    for branch in branches:
        filial_name = branch.get("name", "Филиал не указан")
        categories = branch.get("categories", [])
        for category in categories:
            category_name = category.get("name", "Категория не указана")
            services = category.get("services", [])
            for service in services:
                service_name = service.get("name", "Услуга не указана")
                price = service.get("price", "Цена не указана")
                service_description = service.get("description", "")
                employees = service.get("employees", [])
                if employees:
                    for emp in employees:
                        employee_full_name = emp.get("full_name", "Специалист не указан")
                        employee_description = emp.get("description", "Описание не указано")
                        record = {
                            "filialName": filial_name,
                            "categoryName": category_name,
                            "serviceName": service_name,
                            "serviceDescription": service_description,
                            "price": price,
                            "employeeFullName": employee_full_name,
                            "employeeDescription": employee_description
                        }
                        records.append(record)
                else:
                    record = {
                        "filialName": filial_name,
                        "categoryName": category_name,
                        "serviceName": service_name,
                        "serviceDescription": service_description,
                        "price": price,
                        "employeeFullName": "Специалист не указан",
                        "employeeDescription": "Описание не указано"
                    }
                    records.append(record)
    return records


async def prepare_data(tenant_id: str):
    """
    Подготавливает данные для тенанта: загружает JSON, строит эмбеддинги, BM25 и FAISS-индекс.
    """
    tenant_path = get_tenant_path(tenant_id)
    data_file = tenant_path / "data.json"
    embeddings_file = tenant_path / "embeddings.npy"
    bm25_file = tenant_path / "bm25.pkl"
    faiss_index_file = tenant_path / "faiss_index.index"
    
    if all([f.exists() for f in [data_file, embeddings_file, bm25_file, faiss_index_file]]):
        file_age = time.time() - os.path.getmtime(data_file)
        if file_age < 2_592_000:
            async with aiofiles.open(data_file, "r", encoding="utf-8") as f:
                data = json.loads(await f.read())
            embeddings = np.load(embeddings_file)
            with open(bm25_file, "rb") as f:
                bm25 = pickle.load(f)
            index = faiss.read_index(str(faiss_index_file))
            return data, embeddings, bm25, index

    records = await load_json_data(tenant_id)
    documents = [extract_text_fields(record) for record in records]

    loop = asyncio.get_event_loop()
    embeddings, bm25 = await asyncio.gather(
        loop.run_in_executor(None, lambda: search_model.encode(documents, convert_to_tensor=True).cpu().numpy()),
        loop.run_in_executor(None, lambda: BM25Okapi([tokenize_text(doc) for doc in documents]))
    )

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    faiss.write_index(index, str(faiss_index_file))

    async with aiofiles.open(data_file, "w", encoding="utf-8") as f:
        await f.write(json.dumps({
            "records": records,
            "raw_texts": documents,
            "timestamp": time.time()
        }, ensure_ascii=False, indent=4))

    np.save(embeddings_file, embeddings)
    with open(bm25_file, "wb") as f:
        pickle.dump(bm25, f)

    return {"records": records, "raw_texts": documents}, embeddings, bm25, index


async def update_json_file(mydtoken: str, tenant_id: str):
    """
    Обновляет данные, запрашивая страницы последовательно, но ограничиваясь первыми 50 страницами.
    """
    tenant_path = get_tenant_path(tenant_id)
    file_path = os.path.join(BASE_DIR, f"{tenant_id}.json")
    
    if os.path.exists(file_path):
        file_age = time.time() - os.path.getmtime(file_path)
        if file_age < 2_592_000:
            logger.info(f"Файл {file_path} актуален, пропускаем обновление.")
            return
    
    for f in tenant_path.glob("*"):
        try:
            os.remove(f)
        except Exception as e:
            logger.error(f"Ошибка удаления файла {f}: {e}")
    
    try:
        async with aiohttp.ClientSession() as session:
            headers = {"Authorization": f"Bearer {mydtoken}"}
            params = {"tenantId": tenant_id, "page": 1}
            all_data = []
            max_pages = 500
            while True:
                if params["page"] > max_pages:
                    logger.info(f"Достигнут лимит {max_pages} страниц, завершаем загрузку.")
                    break
                async with session.get(API_URL, headers=headers, params=params) as response:
                    response.raise_for_status()
                    data = await response.json()
                    branches = data.get("data", {}).get("branches", [])
                    if not branches:
                        logger.info(f"Страница {params['page']} пустая, завершаем загрузку.")
                        break
                    all_data.extend(branches)
                    logger.info(f"Получено {len(branches)} записей с страницы {params['page']}.")
                    params["page"] += 1

            logger.info(f"Общее число полученных филиалов: {len(all_data)}")
            async with aiofiles.open(file_path, "w", encoding="utf-8") as json_file:
                await json_file.write(json.dumps(
                    {"code": data.get("code", 200), "data": {"branches": all_data}},
                    ensure_ascii=False,
                    indent=4
                ))
    except Exception as e:
        logger.error(f"Ошибка при обновлении файла: {str(e)}")
        raise HTTPException(status_code=500, detail="Ошибка обновления данных.")


async def rerank_with_cross_encoder(query: str, candidates: List[int], raw_texts: List[str]) -> List[int]:
    """Реранкинг топ-10 кандидатов с использованием кросс-энкодера"""
    cross_inp = [(query, raw_texts[idx]) for idx in candidates]
    loop = asyncio.get_event_loop()
    cross_scores = await loop.run_in_executor(None, lambda: cross_encoder.predict(cross_inp))
    sorted_indices = np.argsort(cross_scores)[::-1].tolist()
    return [candidates[i] for i in sorted_indices]


async def get_free_time_slots(employee_id: str, service_ids: List[str], date_time: str, 
                           tenant_id: str = DEFAULT_TENANT_ID, filial_id: str = None, 
                           lang_id: str = DEFAULT_LANG_ID) -> dict:
    """
    Вызывает API для получения свободных временных слотов сотрудника.
    
    Args:
        employee_id: ID сотрудника
        service_ids: Список ID услуг
        date_time: Дата в формате YYYY-MM-DD
        tenant_id: ID тенанта (по умолчанию используется дефолтный)
        filial_id: ID филиала
        lang_id: Язык (по умолчанию "ru")
        
    Returns:
        dict: Ответ API с доступными временными слотами
    """
    if not filial_id:
        logger.warning("Не указан filial_id для получения свободных слотов")
        return {"status": "error", "message": "Не указан filial_id"}
    
    url = f"{API_BASE_URL}/getFreeTimesOfEmployeeByChoosenServices"
    payload = {
        "employeeId": employee_id,
        "serviceId": service_ids,
        "dateTime": date_time,
        "tenantId": tenant_id,
        "filialId": filial_id,
        "langId": lang_id
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Ошибка API getFreeTimesOfEmployeeByChoosenServices: {response.status}, {error_text}")
                    return {"status": "error", "message": f"Ошибка при получении свободных слотов: {response.status}"}
                
                data = await response.json()
                logger.info(f"Получены свободные слоты для сотрудника {employee_id}: {len(data)} слотов")
                
                # Преобразуем данные в более удобный формат
                formatted_slots = []
                for slot in data:
                    start_time = slot.get("startTime", "")
                    end_time = slot.get("endTime", "")
                    formatted_slots.append({
                        "start_time": start_time,
                        "end_time": end_time,
                        "slot": f"{start_time}-{end_time}"
                    })
                
                return {
                    "status": "success",
                    "free_slots": formatted_slots,
                    "date": date_time,
                    "employee_id": employee_id
                }
    except Exception as e:
        logger.error(f"Ошибка при вызове API getFreeTimesOfEmployeeByChoosenServices: {e}")
        return {"status": "error", "message": f"Произошла ошибка: {str(e)}"}


async def add_record(client_phone: str, service_id: str, filial_id: str, 
                   date_of_record: str, start_time: str, end_time: str, 
                   employee_id: str, service_name: str = None, price: float = 0, 
                   tenant_id: str = DEFAULT_TENANT_ID, lang_id: str = DEFAULT_LANG_ID) -> dict:
    """
    Вызывает API для создания записи клиента на прием.
    
    Args:
        client_phone: Номер телефона клиента
        service_id: ID услуги
        filial_id: ID филиала
        date_of_record: Дата записи в формате YYYY-MM-DD
        start_time: Время начала в формате HH:MM
        end_time: Время окончания в формате HH:MM
        employee_id: ID сотрудника
        service_name: Название услуги (опционально)
        price: Цена услуги (опционально)
        tenant_id: ID тенанта (по умолчанию используется дефолтный)
        lang_id: Язык (по умолчанию "ru")
        
    Returns:
        dict: Ответ API с результатом создания записи
    """
    url = f"{API_BASE_URL}/addRecord"
    
    # Вычисляем продолжительность услуги в минутах
    start_hour, start_min = map(int, start_time.split(':'))
    end_hour, end_min = map(int, end_time.split(':'))
    start_minutes = start_hour * 60 + start_min
    end_minutes = end_hour * 60 + end_min
    duration = end_minutes - start_minutes
    
    payload = {
        "langId": lang_id,
        "clientPhoneNumber": client_phone,
        "services": [
            {
                "rowNumber": 0,
                "parentId": "",
                "serviceId": service_id,
                "serviceName": service_name or "Услуга",
                "countService": 1,
                "discount": 0,
                "price": price,
                "salePrice": price,
                "complexServiceId": "",
                "durationService": duration
            }
        ],
        "filialId": filial_id,
        "dateOfRecord": date_of_record,
        "startTime": start_time,
        "endTime": end_time,
        "durationOfTime": duration,
        "colorCodeRecord": DEFAULT_COLOR_CODE,
        "toEmployeeId": employee_id,
        "totalPrice": price,
        "trafficChannel": DEFAULT_TRAFFIC_CHANNEL,
        "trafficChannelId": ""
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Ошибка API addRecord: {response.status}, {error_text}")
                    return {"status": "error", "message": f"Ошибка при создании записи: {response.status}"}
                
                data = await response.json()
                logger.info(f"Создана запись: {data}")
                
                # Обработка ответа API
                return {
                    "status": "success",
                    "record_id": data.get("id", ""),
                    "message": "Запись успешно создана",
                    "date": date_of_record,
                    "time": f"{start_time}-{end_time}"
                }
    except Exception as e:
        logger.error(f"Ошибка при вызове API addRecord: {e}")
        return {"status": "error", "message": f"Произошла ошибка: {str(e)}"}


async def search_services_by_category(category_name: str, tenant_id: str = DEFAULT_TENANT_ID) -> dict:
    """
    Поиск услуг по категории через API.
    
    Args:
        category_name: Название категории
        tenant_id: ID тенанта (по умолчанию используется дефолтный)
        
    Returns:
        dict: Список услуг в категории
    """
    url = f"{API_BASE_URL}/search/{category_name}"
    
    try:
        async with aiohttp.ClientSession() as session:
            params = {"tenantId": tenant_id}
            async with session.get(url, params=params) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Ошибка API search: {response.status}, {error_text}")
                    return {"status": "error", "message": f"Ошибка при поиске услуг: {response.status}"}
                
                data = await response.json()
                logger.info(f"Найдено услуг в категории {category_name}: {len(data)}")
                
                return {
                    "status": "success",
                    "services": data,
                    "category": category_name
                }
    except Exception as e:
        logger.error(f"Ошибка при вызове API search: {e}")
        return {"status": "error", "message": f"Произошла ошибка: {str(e)}"}


free_times_function = {
    "type": "function",
    "function": {
        "name": "get_free_time_slots",
        "description": "Получает свободные временные слоты для записи к специалисту",
        "parameters": {
            "type": "object",
            "properties": {
                "employee_id": {
                    "type": "string",
                    "description": "ID сотрудника (специалиста)"
                },
                "service_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Список ID услуг"
                },
                "date_time": {
                    "type": "string",
                    "description": "Дата в формате YYYY-MM-DD"
                },
                "filial_id": {
                    "type": "string",
                    "description": "ID филиала"
                }
            },
            "required": ["employee_id", "service_ids", "date_time", "filial_id"]
        }
    }
}

add_record_function = {
    "type": "function",
    "function": {
        "name": "add_record",
        "description": "Создает запись клиента на прием к специалисту",
        "parameters": {
            "type": "object",
            "properties": {
                "client_phone": {
                    "type": "string",
                    "description": "Номер телефона клиента"
                },
                "service_id": {
                    "type": "string",
                    "description": "ID услуги"
                },
                "filial_id": {
                    "type": "string",
                    "description": "ID филиала"
                },
                "date_of_record": {
                    "type": "string",
                    "description": "Дата записи в формате YYYY-MM-DD"
                },
                "start_time": {
                    "type": "string",
                    "description": "Время начала в формате HH:MM"
                },
                "end_time": {
                    "type": "string",
                    "description": "Время окончания в формате HH:MM"
                },
                "employee_id": {
                    "type": "string",
                    "description": "ID сотрудника"
                },
                "service_name": {
                    "type": "string",
                    "description": "Название услуги (опционально)"
                },
                "price": {
                    "type": "number",
                    "description": "Цена услуги (опционально)"
                }
            },
            "required": ["client_phone", "service_id", "filial_id", "date_of_record", "start_time", "end_time", "employee_id"]
        }
    }
}

search_services_function = {
    "type": "function",
    "function": {
        "name": "search_services_by_category",
        "description": "Ищет услуги по названию категории",
        "parameters": {
            "type": "object",
            "properties": {
                "category_name": {
                    "type": "string",
                    "description": "Название категории услуг"
                }
            },
            "required": ["category_name"]
        }
    }
}

async def generate_yandexgpt_response(user_question, user_id, context=None):
    """Генерация ответа с использованием YandexGPT и функций модели."""
    
    # Инициализация SDK для Yandex Cloud if not already done
    sdk = YCloudML()
    
    # Формирование системного промпта
    system_prompt = """Вы - ассистент медицинской клиники. Вы помогаете клиентам записаться к специалистам, отвечаете на вопросы об услугах и ценах.
Отвечайте коротко, вежливо и по существу. 
Если вам нужно записать клиента на прием, получите его телефон, выберите услугу, врача, филиал и время для записи.
Используйте имеющиеся функции для получения информации и создания записей."""

    if user_id not in conversation_history:
        conversation_history[user_id] = []
    
    # Формирование сообщений для модели
    messages = [{"role": "system", "content": system_prompt}]
    
    # Добавляем контекст, если он есть
    if context and len(context) > 0:
        messages.append({"role": "system", "content": f"Вот информация, которая может быть полезна: {context}"})
    
    # Добавляем историю диалога
    for entry in conversation_history[user_id]:
        messages.append({"role": entry["role"], "content": entry["content"]})
    
    # Добавляем текущий вопрос пользователя
    messages.append({"role": "user", "content": user_question})
    
    # Засекаем время выполнения запроса
    start_time = time.time()
    
    try:
        # Делаем запрос к модели с функциями
        response = await sdk.chat.completions.create(
            model="yandexgpt",
            messages=messages,
            max_tokens=1000,
            temperature=0.7,
            tools=[free_times_function, add_record_function, search_services_function]
        )
        
        # Получаем ответ от модели
        response_message = response.choices[0].message
        
        # Обрабатываем функции, если они вызваны
        if hasattr(response_message, 'tool_calls') and response_message.tool_calls:
            # Логируем вызов функции
            logger.info(f"Модель вызвала функцию: {response_message.tool_calls}")
            
            # Обрабатываем каждый вызов функции
            for tool_call in response_message.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                
                function_result = None
                
                # Вызываем соответствующую функцию
                if function_name == "get_free_time_slots":
                    function_result = await get_free_time_slots(
                        employee_id=function_args.get("employee_id"),
                        service_ids=function_args.get("service_ids"),
                        date_time=function_args.get("date_time"),
                        filial_id=function_args.get("filial_id")
                    )
                elif function_name == "add_record":
                    function_result = await add_record(
                        client_phone=function_args.get("client_phone"),
                        service_id=function_args.get("service_id"),
                        filial_id=function_args.get("filial_id"),
                        date_of_record=function_args.get("date_of_record"),
                        start_time=function_args.get("start_time"),
                        end_time=function_args.get("end_time"),
                        employee_id=function_args.get("employee_id"),
                        service_name=function_args.get("service_name"),
                        price=function_args.get("price", 0)
                    )
                elif function_name == "search_services_by_category":
                    function_result = await search_services_by_category(
                        category_name=function_args.get("category_name")
                    )
                
                # Добавляем результат функции в сообщения
                if function_result:
                    messages.append({
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": tool_call.id,
                                "type": "function",
                                "function": {
                                    "name": function_name,
                                    "arguments": tool_call.function.arguments
                                }
                            }
                        ]
                    })
                    
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(function_result, ensure_ascii=False)
                    })
            
            # Отправляем второй запрос с результатами функции
            second_response = await sdk.chat.completions.create(
                model="yandexgpt",
                messages=messages,
                max_tokens=1000,
                temperature=0.7
            )
            
            # Получаем финальный ответ от модели
            final_response = second_response.choices[0].message.content
            
            # Добавляем ответ в историю диалога
            conversation_history[user_id].append({"role": "user", "content": user_question})
            conversation_history[user_id].append({"role": "assistant", "content": final_response})
            
            # Ограничиваем историю до последних 10 сообщений
            if len(conversation_history[user_id]) > 20:
                conversation_history[user_id] = conversation_history[user_id][-20:]
            
            logger.info(f"Получен итоговый ответ после вызова функций за {time.time() - start_time:.2f} сек")
            return final_response
        else:
            # Обычный ответ без вызова функций
            assistant_response = response_message.content
            
            # Добавляем ответ в историю диалога
            conversation_history[user_id].append({"role": "user", "content": user_question})
            conversation_history[user_id].append({"role": "assistant", "content": assistant_response})
            
            # Ограничиваем историю до последних 10 сообщений
            if len(conversation_history[user_id]) > 20:
                conversation_history[user_id] = conversation_history[user_id][-20:]
            
            logger.info(f"Получен стандартный ответ за {time.time() - start_time:.2f} сек")
            return assistant_response
    
    except Exception as e:
        logger.error(f"Ошибка при генерации ответа: {e}")
        return f"Произошла ошибка при обработке запроса: {str(e)}"


@app.post("/ask")
async def ask_assistant(
    user_id: str = Form(...),
    question: Optional[str] = Form(None),
    mydtoken: str = Form(...),
    tenant_id: str = Form(...),
    file: UploadFile = File(None)
):
    try:
        # Очистка устаревших диалогов
        current_time = time.time()
        expired_users = [uid for uid, data in conversation_history.items() if current_time - data["last_active"] > 22296]
        for uid in expired_users:
            del conversation_history[uid]
            logger.info(f"Удалена история диалога для {uid} (устарела)")

        recognized_text = None
        if file and file.filename:
            temp_path = f"/tmp/{file.filename}"
            try:
                async with aiofiles.open(temp_path, "wb") as temp_file:
                    await temp_file.write(await file.read())
                loop = asyncio.get_event_loop()
                recognized_text = await loop.run_in_executor(None, lambda: recognize_audio_with_sdk(temp_path))
                logger.info(f"Распознан текст из аудио: {recognized_text}")
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            if not recognized_text:
                raise HTTPException(status_code=500, detail="Ошибка распознавания речи из файла.")

        input_text = recognized_text or question
        if not input_text:
            raise HTTPException(status_code=400, detail="Необходимо передать текст или файл.")

        # Обновляем базовый JSON-файл при необходимости
        await update_json_file(mydtoken, tenant_id)
        
        # Проверяем наличие файла data.json и при необходимости конвертируем
        tenant_path = get_tenant_path(tenant_id)
        data_json_path = tenant_path / "data.json"
        if not data_json_path.exists():
            logger.info(f"Файл data.json не найден, создаем из базового файла")
            conversion_success = await convert_base_json_to_data_json(tenant_id)
            if not conversion_success:
                raise HTTPException(status_code=404, detail=f"Не удалось подготовить данные для поиска")
        
        # Подготавливаем данные для поиска
        data_dict, embeddings, bm25, faiss_index = await prepare_data(tenant_id)
        normalized_question = normalize_text(input_text)
        tokenized_query = tokenize_text(normalized_question)
        
        # BM25 поиск
        bm25_scores = bm25.get_scores(tokenized_query)
        top_bm25_indices = np.argsort(bm25_scores)[::-1][:50].tolist()
        
        # Векторный поиск
        loop = asyncio.get_event_loop()
        query_embedding = await loop.run_in_executor(
            None,
            lambda: search_model.encode(normalized_question, convert_to_tensor=True).cpu().numpy()
        )
        D, I = faiss_index.search(query_embedding.reshape(1, -1), 50)
        DISTANCE_THRESHOLD = 1.0
        filtered_faiss = [idx for idx, dist in zip(I[0].tolist(), D[0].tolist()) if dist < DISTANCE_THRESHOLD]
        if not filtered_faiss:
            filtered_faiss = I[0].tolist()
        top_faiss_indices = filtered_faiss
        
        # Объединяем результаты и ранжируем с помощью CrossEncoder
        combined_indices = list(set(top_bm25_indices + top_faiss_indices))[:50]
        top_10_indices = await rerank_with_cross_encoder(
            query=normalized_question,
            candidates=combined_indices[:30],
            raw_texts=data_dict["raw_texts"]
        )
        
        # Формируем контекст в формате markdown для лучшей читаемости
        context = "\n\n".join([
            f"**Документ {i+1}:**\n" 
            f"* Филиал: {data_dict['records'][idx].get('filialName', 'Не указан')}\n"
            f"* Категория: {data_dict['records'][idx].get('categoryName', 'Не указана')}\n"
            f"* Услуга: {data_dict['records'][idx].get('serviceName', 'Не указана')}\n"
            f"* Цена: {data_dict['records'][idx].get('price', 'Цена не указана')} руб.\n"
            f"* Специалист: {data_dict['records'][idx].get('employeeFullName', 'Не указан')}\n"
            f"* Описание: {data_dict['records'][idx].get('employeeDescription', 'Описание не указано')}"  
            for i, idx in enumerate(top_10_indices[:5])
        ])
        
        # Инициализируем историю диалога для нового пользователя
        if user_id not in conversation_history:
            conversation_history[user_id] = {"history": [], "last_active": time.time(), "greeted": False}

        # Обновляем время последней активности
        conversation_history[user_id]["last_active"] = time.time()
        
        # Генерируем ответ через YandexGPT
        response_text = await generate_yandexgpt_response(input_text, user_id, context)
        
        # Сохраняем историю диалога
        conversation_history[user_id]["history"].append({
            "user_query": input_text,
            "assistant_response": response_text,
            "search_results": [data_dict['records'][idx] for idx in top_10_indices]
        })

        return {"response": response_text}
    except Exception as e:
        logger.error(f"Ошибка обработки запроса: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ошибка обработки запроса: {str(e)}")


if __name__ == "__main__":
    logger.info("Запуск сервера на порту 8001...")
    uvicorn.run(app, host="0.0.0.0", port=8001)
