#python:veda-backend/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes import hymns, search, analysis, advanced_features

app = FastAPI(title="Enhanced Rig Veda API", description="API for accessing Rig Veda data with AI and visualization features")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(hymns.router)
app.include_router(analysis.router)
app.include_router(advanced_features.router)

@app.get("/")
async def root():
    return {"message": "Welcome to the Enhanced Rig Veda API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


#2. Advanced Features Router:

#python:veda-backend/routes/advanced_features.py
from fastapi import APIRouter, Depends, BackgroundTasks, Query, HTTPException
from fastapi.responses import JSONResponse
from services.hymn_service import get_hymn_by_id
from services.openai_service import generate_vedic_quiz, compare_hymns, generate_meditation, ask_question
from utils.security import get_api_key
from models.schemas import BackgroundTask
import random
from services.openai_service import call_openai_api

router = APIRouter()
background_tasks = {}

async def run_background_task(task_id: str, func, *args, **kwargs):
    try:
        result = await func(*args, **kwargs)
        background_tasks[task_id] = BackgroundTask(id=task_id, status="completed", result=result)
    except Exception as e:
        background_tasks[task_id] = BackgroundTask(id=task_id, status="failed", result={"error": str(e)})

@router.get("/task/{task_id}")
async def get_task_status(task_id: str):
    task = background_tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task

@router.get("/vedic-quiz")
async def generate_vedic_quiz_route(background_tasks: BackgroundTasks, num_questions: int = Query(5, ge=1, le=10), api_key: str = Depends(get_api_key)):
    task_id = f"quiz_{random.randint(1000, 9999)}"
    background_tasks.add_task(run_background_task, task_id, generate_vedic_quiz, num_questions)
    return {"task_id": task_id}

@router.get("/comparative-analysis")
async def compare_hymns_route(background_tasks: BackgroundTasks, mandala1: int, hymn1: int, mandala2: int, hymn2: int, api_key: str = Depends(get_api_key)):
    task_id = f"compare_{random.randint(1000, 9999)}"
    background_tasks.add_task(run_background_task, task_id, compare_hymns, mandala1, hymn1, mandala2, hymn2)
    return {"task_id": task_id}

@router.get("/generate-meditation")
async def generate_meditation_route(background_tasks: BackgroundTasks, mandala: int, hymn: int, duration: int = Query(10, ge=5, le=30), api_key: str = Depends(get_api_key)):
    task_id = f"meditation_{random.randint(1000, 9999)}"
    background_tasks.add_task(run_background_task, task_id, generate_meditation, mandala, hymn, duration)
    return {"task_id": task_id}

@router.get("/ask-question")
async def ask_question_route(question: str, api_key: str = Depends(get_api_key)):
    answer = await ask_question(question)
    return JSONResponse(content={"question": question, "answer": answer})

@router.get("/test-openai-connection")
async def test_openai_connection():
    try:
        response = await call_openai_api([{"role": "user", "content": "Hello"}])
        return {"status": "success", "message": response}
    except Exception as e:
        return {"status": "error", "message": str(e)}
#

#3. OpenAI Service:

#python:veda-backend/services/openai_service.py
import json
import random
from typing import List
from models.schemas import Hymn, VedicQuiz
from config import OPENAI_API_KEY
import openai

openai.api_key = OPENAI_API_KEY

async def call_openai_api(messages):
    response = await openai.ChatCompletion.acreate(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=1000,
        n=1,
        stop=None,
        temperature=0.7,
    )
    return response.choices[0].message.content.strip()

async def explain_hymn(hymn: Hymn) -> str:
    hymn_text = f"Rig Veda Hymn {hymn.mandala}.{hymn.hymn_number}\n\n"
    hymn_text += f"Title: {hymn.title}\n\n"
    for verse in hymn.verses:
        hymn_text += f"Verse {verse.number}:\n"
        hymn_text += f"Sanskrit: {verse.sanskrit}\n"
        hymn_text += f"Transliteration: {verse.transliteration}\n"
        hymn_text += f"Translation: {verse.translation}\n\n"
    
    prompt = f"Please explain the following hymn from the Rig Veda, including its context, meaning, and significance:\n\n{hymn_text}"
    
    return await call_openai_api([
        {"role": "system", "content": "You are a knowledgeable expert on Vedic literature, especially the Rig Veda."},
        {"role": "user", "content": prompt},
    ])

async def generate_vedic_quiz(num_questions: int) -> List[VedicQuiz]:
    with open('data/rig_veda_formatted_combined.json') as f:
        RIG_VEDA_DATA = json.load(f)
    all_verses = [verse for mandala in RIG_VEDA_DATA['mandalas'] for hymn in mandala['hymns'] for verse in hymn['verses']]
    selected_verses = random.sample(all_verses, num_questions)
    
    quiz_questions = []
    for verse in selected_verses:
        prompt = f"Create a multiple-choice question based on this Rig Veda verse:\n\n{verse['translation']}\n\nProvide four options and indicate the correct answer."
        
        response = await call_openai_api([
            {"role": "system", "content": "You are an expert in creating educational quizzes about Vedic literature."},
            {"role": "user", "content": prompt}
        ])
        
        lines = response.split('\n')
        question = lines[0]
        options = lines[1:5]
        correct_answer = int(lines[5].split()[-1]) - 1
        
        quiz_questions.append(VedicQuiz(
            question=question,
            options=options,
            correct_answer=correct_answer
        ))
    
    return quiz_questions

async def compare_hymns(hymn1: Hymn, hymn2: Hymn) -> str:
    hymn_text1 = "\n".join([verse.translation for verse in hymn1.verses])
    hymn_text2 = "\n".join([verse.translation for verse in hymn2.verses])
    
    prompt = f"Compare and contrast the following two Rig Veda hymns:\n\nHymn 1:\n{hymn_text1}\n\nHymn 2:\n{hymn_text2}\n\nAnalyze their themes, style, and significance."
    
    return await call_openai_api([
        {"role": "system", "content": "You are an expert in comparative analysis of Vedic literature."},
        {"role": "user", "content": prompt}
    ])

async def generate_meditation(hymn: Hymn, duration: int) -> str:
    hymn_text = "\n".join([verse.translation for verse in hymn.verses])
    
    prompt = f"Create a guided meditation script based on the following Rig Veda hymn. The meditation should last approximately {duration} minutes:\n\n{hymn_text}"
    
    return await call_openai_api([
        {"role": "system", "content": "You are an expert in creating guided meditations based on ancient wisdom."},
        {"role": "user", "content": prompt}
    ])

async def ask_question(question: str) -> str:
    prompt = f"Answer the following question about the Rig Veda:\n\n{question}"
    
    return await call_openai_api([
        {"role": "system", "content": "You are an expert on the Rig Veda and Vedic literature."},
        {"role": "user", "content": prompt}
    ])
#

#4. Hymn Service:

#python:veda-backend/services/hymn_service.py
import json
from models.schemas import Hymn, Verse

with open('data/rig_veda_formatted_combined.json', 'r', encoding='utf-8') as f:
    RIG_VEDA_DATA = json.load(f)

def get_all_hymns():
    hymns = []
    for mandala in RIG_VEDA_DATA['mandalas']:
        for hymn in mandala['hymns']:
            hymns.append(Hymn(
                mandala=mandala['number'],
                hymn_number=hymn['number'],
                title=hymn['title'],
                verses=[Verse(**verse) for verse in hymn['verses']]
            ))
    return hymns

def get_hymns_by_mandala(mandala: int):
    return [hymn for hymn in get_all_hymns() if hymn.mandala == mandala]

def get_hymn_by_id(mandala: int, hymn_number: int):
    for hymn in get_all_hymns():
        if hymn.mandala == mandala and hymn.hymn_number == hymn_number:
            return hymn
    return None

def search_verses(query: str, search_translation: bool = False):
    matching_hymns = []
    for hymn in get_all_hymns():
        matching_verses = []
        for verse in hymn.verses:
            if search_translation:
                if query.lower() in verse.translation.lower():
                    matching_verses.append(verse)
            else:
                if query.lower() in verse.sanskrit.lower() or query.lower() in verse.transliteration.lower():
                    matching_verses.append(verse)
        if matching_verses:
            matching_hymns.append(Hymn(
                mandala=hymn.mandala,
                hymn_number=hymn.hymn_number,
                title=hymn.title,
                verses=matching_verses
            ))
    return matching_hymns
#

#
#5. Data Preparation Script:

#python:veda-backend/data/test.py
import json
import re
from pydub import AudioSegment
import numpy as np

def prepare_mantra_healing_dataset():
    with open('rig_veda_formatted_combined.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    mantra_data = []

    for mandala in data['mandalas']:
        for hymn in mandala['hymns']:
            for verse in hymn['verses']:
                mantras = re.findall(r'\b[A-Z]+\b', verse['transliteration'])
                if mantras:
                    audio_file = f"audio_data/{mandala['number']}/{hymn['number']}.mp3"
                    try:
                        audio = AudioSegment.from_mp3(audio_file)
                        audio_array = np.array(audio.get_array_of_samples())
                        frequency = np.abs(np.fft.fft(audio_array)[:len(audio_array)//2])
                        dominant_freq = np.argmax(frequency)
                        
                        mantra_data.append({
                            'mantra': mantras[0],
                            'sanskrit': verse['sanskrit'],
                            'translation': verse['translation'],
                            'dominant_frequency': float(dominant_freq),
                            'audio_length': len(audio),
                            'mandala': mandala['number'],
                            'hymn': hymn['number'],
                            'verse': verse['number']
                        })
                    except FileNotFoundError:
                        print(f"Audio file not found: {audio_file}")

    with open('mantra_healing_dataset.json', 'w', encoding='utf-8') as f:
        json.dump(mantra_data, f, ensure_ascii=False, indent=2)

    print("Mantra sound healing dataset prepared. Check 'mantra_healing_dataset.json' for results.")

if __name__ == "__main__":
    prepare_mantra_healing_dataset()
