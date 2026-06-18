import os
import streamlit as st
import torch
import numpy as np
from PIL import Image
import pandas as pd
import time
import json
import threading
import io

from models.generator import Generator
from models.text_encoder import TextEncoder
from utils.visualization import tensor_to_pil, generate_3d_viewer_html, pil_to_base64
from utils.samples import find_latest_sample_epoch
from train import train
from utils.data_prep import generate_test_dataset
from data.dataset import MinecraftSkinDataset

# Set page configuration
st.set_page_config(
    page_title="Minecraft Skin AI Painter",
    page_icon="🎨",
    layout="wide"
)

# Sidebar - System Status
st.sidebar.title("🎨 Minecraft Skin AI")
st.sidebar.write("Нейросеть для генерации скинов Майнкрафт по текстовому описанию.")

# Check for CUDA/GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
st.sidebar.success(f"Вычисления: **{device.upper()}**")

# Define default paths
DEFAULT_DATA_DIR = "dataset"
DEFAULT_CHECKPOINT_DIR = "checkpoints"
DEFAULT_SAMPLES_DIR = "samples"

# Initialize folders
os.makedirs(DEFAULT_DATA_DIR, exist_ok=True)
os.makedirs(DEFAULT_CHECKPOINT_DIR, exist_ok=True)
os.makedirs(DEFAULT_SAMPLES_DIR, exist_ok=True)

STATUS_FILE = os.path.join(DEFAULT_CHECKPOINT_DIR, "training_status.json")

def get_default_status():
    return {
        "status": "idle",
        "current_epoch": 0,
        "total_epochs": 150,
        "g_loss": 0.0,
        "d_loss": 0.0,
        "critic_real": 0.0,
        "critic_fake": 0.0,
        "gradient_penalty": 0.0,
        "generator_updates": 0,
        "loss_history": {
            "Epoch": [],
            "Critic Loss": [],
            "Generator Loss": []
        },
        "diagnostics_history": {
            "Epoch": [],
            "Critic Real": [],
            "Critic Fake": [],
            "Gradient Penalty": [],
            "Generator Updates": []
        },
        "error_message": "",
        "stop_requested": False
    }

def load_status():
    status = get_default_status()
    if os.path.exists(STATUS_FILE):
        try:
            with open(STATUS_FILE, "r", encoding="utf-8") as f:
                loaded_status = json.load(f)
            status.update(loaded_status)
            status["loss_history"] = {**get_default_status()["loss_history"], **loaded_status.get("loss_history", {})}
            status["diagnostics_history"] = {**get_default_status()["diagnostics_history"], **loaded_status.get("diagnostics_history", {})}
        except Exception:
            pass
    return status

def save_status(status_dict):
    try:
        with open(STATUS_FILE, "w", encoding="utf-8") as f:
            json.dump(status_dict, f, indent=4, ensure_ascii=False)
    except Exception as e:
        print(f"Error saving status: {e}")

def run_training_thread(data_dir, checkpoint_dir, samples_dir, epochs, batch_size, g_lr, d_lr, n_critic, gp_lambda, resume, device):
    status = load_status()
    previous_loss_history = status.get("loss_history", get_default_status()["loss_history"])
    previous_diagnostics_history = status.get("diagnostics_history", get_default_status()["diagnostics_history"])
    previous_epochs = previous_loss_history.get("Epoch", [])
    epoch_offset = max(previous_epochs) if resume and previous_epochs else 0
    status["status"] = "training"
    status["current_epoch"] = epoch_offset
    status["total_epochs"] = epoch_offset + epochs
    status["g_loss"] = 0.0
    status["d_loss"] = 0.0
    status["critic_real"] = 0.0
    status["critic_fake"] = 0.0
    status["gradient_penalty"] = 0.0
    status["generator_updates"] = 0
    status["loss_history"] = previous_loss_history if resume else {
            "Epoch": [],
            "Critic Loss": [],
            "Generator Loss": []
        }
    status["diagnostics_history"] = previous_diagnostics_history if resume else {
            "Epoch": [],
            "Critic Real": [],
            "Critic Fake": [],
            "Gradient Penalty": [],
            "Generator Updates": []
        }
    status["error_message"] = ""
    status["stop_requested"] = False
    save_status(status)
    
    def thread_callback(epoch, d_loss, g_loss, diagnostics=None):
        curr_status = load_status()
        if curr_status.get("stop_requested", False):
            curr_status["status"] = "idle"
            curr_status["stop_requested"] = False
            save_status(curr_status)
            return False
        
        diagnostics = diagnostics or {}
        critic_real = diagnostics.get("critic_real", 0.0)
        critic_fake = diagnostics.get("critic_fake", 0.0)
        gradient_penalty = diagnostics.get("gradient_penalty", 0.0)
        generator_updates = diagnostics.get("generator_updates", 0)
        
        curr_status["current_epoch"] = epoch
        curr_status["d_loss"] = d_loss
        curr_status["g_loss"] = g_loss
        curr_status["critic_real"] = critic_real
        curr_status["critic_fake"] = critic_fake
        curr_status["gradient_penalty"] = gradient_penalty
        curr_status["generator_updates"] = generator_updates
        curr_status["loss_history"]["Epoch"].append(epoch)
        curr_status["loss_history"]["Critic Loss"].append(d_loss)
        curr_status["loss_history"]["Generator Loss"].append(g_loss)
        curr_status["diagnostics_history"]["Epoch"].append(epoch)
        curr_status["diagnostics_history"]["Critic Real"].append(critic_real)
        curr_status["diagnostics_history"]["Critic Fake"].append(critic_fake)
        curr_status["diagnostics_history"]["Gradient Penalty"].append(gradient_penalty)
        curr_status["diagnostics_history"]["Generator Updates"].append(generator_updates)
        save_status(curr_status)
        return True

    try:
        train(
            data_dir=data_dir,
            checkpoint_dir=checkpoint_dir,
            samples_dir=samples_dir,
            epochs=epochs,
            batch_size=batch_size,
            g_lr=g_lr,
            d_lr=d_lr,
            n_critic=n_critic,
            gp_lambda=gp_lambda,
            resume=resume,
            epoch_offset=epoch_offset,
            device=device,
            callbacks=thread_callback
        )
        
        # After successful completion, check if it wasn't stopped
        final_status = load_status()
        if final_status["status"] == "training":
            final_status["status"] = "completed"
            save_status(final_status)
            
    except Exception as e:
        import traceback
        err_msg = f"{str(e)}\n{traceback.format_exc()}"
        final_status = load_status()
        final_status["status"] = "error"
        final_status["error_message"] = err_msg
        save_status(final_status)

# Helper function to check model availability
def is_model_available():
    candidates = [
        os.path.join(DEFAULT_CHECKPOINT_DIR, "generator_ema_latest.pth"),
        os.path.join(DEFAULT_CHECKPOINT_DIR, "generator_ema_final.pth"),
        os.path.join(DEFAULT_CHECKPOINT_DIR, "generator_latest.pth"),
        os.path.join(DEFAULT_CHECKPOINT_DIR, "generator_final.pth"),
    ]
    return any(os.path.exists(p) for p in candidates)

def is_training_checkpoint_available():
    generator_latest_path = os.path.join(DEFAULT_CHECKPOINT_DIR, "generator_latest.pth")
    generator_final_path = os.path.join(DEFAULT_CHECKPOINT_DIR, "generator_final.pth")
    discriminator_latest_path = os.path.join(DEFAULT_CHECKPOINT_DIR, "discriminator_latest.pth")
    discriminator_final_path = os.path.join(DEFAULT_CHECKPOINT_DIR, "discriminator_final.pth")
    has_generator = os.path.exists(generator_latest_path) or os.path.exists(generator_final_path)
    has_discriminator = os.path.exists(discriminator_latest_path) or os.path.exists(discriminator_final_path)
    return has_generator and has_discriminator

def get_latest_model_path():
    candidates = [
        os.path.join(DEFAULT_CHECKPOINT_DIR, "generator_ema_latest.pth"),
        os.path.join(DEFAULT_CHECKPOINT_DIR, "generator_ema_final.pth"),
        os.path.join(DEFAULT_CHECKPOINT_DIR, "generator_latest.pth"),
        os.path.join(DEFAULT_CHECKPOINT_DIR, "generator_final.pth"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None

# Load text encoder (cached to prevent reloading)
@st.cache_resource
def get_text_encoder():
    return TextEncoder(device=device)

# Load Generator
@st.cache_resource
def load_generator(weights_path, weights_mtime):
    text_encoder = get_text_encoder()
    model = Generator(latent_dim=128, text_embed_dim=text_encoder.embedding_dim)
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

# Main App Layout
st.title("🎨 Генератор Скинов Майнкрафт")

tabs = st.tabs(["✨ Генерация Скинов", "🏋️ Обучение Нейросети", "📁 Управление Датасетом"])

# ==========================================
# TAB 1: GENERATION
# ==========================================
with tabs[0]:
    st.header("✨ Создать новый скин")
    
    if not is_model_available():
        st.warning("⚠️ Нейросеть еще не обучена. Перейдите во вкладку 'Обучение Нейросети' или загрузите модель.")
    else:
        weights_path = get_latest_model_path()
        weights_mtime = os.path.getmtime(weights_path)
        
        col_input, col_output = st.columns([1, 1.2])
        
        with col_input:
            prompt_input = st.text_input(
                "Опишите скин (на русском или английском):",
                placeholder="Например: крипер в красной толстовке, или: futuristic dark soldier with red eyes",
                help="Мультиязычный энкодер понимает любой язык!"
            )
            
            seed_mode = st.radio("Настройки генерации", ["Случайный seed", "Фиксированный seed"])
            if seed_mode == "Фиксированный seed":
                seed = st.number_input("Seed (число)", value=42, step=1)
                torch.manual_seed(seed)
                np.random.seed(seed)
            
            generate_btn = st.button("🚀 Нарисовать скин", type="primary", width='stretch')
            
        with col_output:
            if generate_btn and prompt_input:
                with st.spinner("🧑‍🎨 Нейросеть рисует ваш скин..."):
                    try:
                        # Load models
                        netG = load_generator(weights_path, weights_mtime)
                        encoder = get_text_encoder()
                        
                        # Process text
                        text_embed = encoder.encode(prompt_input).unsqueeze(0).to(device)
                        
                        # Generate noise
                        z = torch.randn(1, 128, device=device)
                        
                        # Run inference
                        with torch.no_grad():
                            generated_tensor = netG(z, text_embed)[0]
                        
                        # Convert to PIL
                        pil_skin = tensor_to_pil(generated_tensor)
                        
                        # Show result
                        st.subheader("Готовый результат:")
                        
                        col_2d, col_3d = st.columns([1, 1.5])
                        
                        with col_2d:
                            st.write("**Развертка (2D):**")
                            # Resize 2D skin for better visual in Streamlit
                            st.image(pil_skin.resize((192, 192), Image.Resampling.NEAREST), width='content', output_format="PNG")
                            
                            # Download Button
                            img_byte_arr = io.BytesIO()
                            pil_skin.save(img_byte_arr, format='PNG')
                            img_bytes = img_byte_arr.getvalue()
                            
                            st.download_button(
                                label="💾 Скачать .png",
                                data=img_bytes,
                                file_name=f"skin_{prompt_input.replace(' ', '_')}.png",
                                mime="image/png",
                                width='stretch'
                            )
                            
                        with col_3d:
                            st.write("**Интерактивный 3D-просмотр:**")
                            # Embed three.js skin viewer
                            skin_b64 = pil_to_base64(pil_skin)
                            html_code = generate_3d_viewer_html(skin_b64, width=320, height=360)
                            st.components.v1.html(html_code, height=380)
                            
                    except Exception as e:
                        st.error(f"Ошибка при генерации: {e}")
            elif not prompt_input and generate_btn:
                st.error("Пожалуйста, введите описание скина!")

# ==========================================
# TAB 2: TRAINING
# ==========================================
with tabs[1]:
    st.header("🏋️ Обучение нейросети")
    
    # Load current status
    training_status = load_status()
    is_currently_training = (training_status["status"] == "training")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Настройки обучения")
        
        data_dir_input = st.text_input("Папка с картинками датасета:", value=DEFAULT_DATA_DIR, disabled=is_currently_training)
        
        # Check files in dataset folder
        png_files = []
        if os.path.exists(data_dir_input):
            png_files = [f for f in os.listdir(data_dir_input) if f.lower().endswith(".png")]
            
        st.info(f"Найдено скинов в папке: **{len(png_files)}**")
        
        epochs_input = st.number_input("Количество эпох (проходов):", min_value=1, max_value=5000, value=150, step=10, disabled=is_currently_training)
        batch_size_input = st.selectbox("Размер батча (Batch Size):", [4, 8, 16, 32, 64], index=2, disabled=is_currently_training)
        
        g_lr_input = st.number_input("Скорость обучения генератора (G LR):", value=1e-4, format="%.5f", disabled=is_currently_training)
        d_lr_input = st.number_input("Скорость обучения дискриминатора (D LR):", value=5e-5, format="%.5f", disabled=is_currently_training)
        n_critic_input = st.number_input("Шагов критика на шаг генератора:", min_value=1, max_value=10, value=3, step=1, disabled=is_currently_training)
        gp_lambda_input = st.number_input("Сила Gradient Penalty:", min_value=0.0, max_value=50.0, value=10.0, step=1.0, format="%.2f", disabled=is_currently_training)
        resume_available = is_training_checkpoint_available()
        resume_input = st.checkbox("Продолжить с последнего checkpoint", value=resume_available, disabled=is_currently_training or not resume_available)
        if not resume_available:
            st.caption("Checkpoint для продолжения пока не найден.")
        
        if is_currently_training:
            stop_training = st.button("⏹️ Остановить Обучение", width='stretch', type="secondary")
            if stop_training:
                training_status["stop_requested"] = True
                save_status(training_status)
                st.warning("Запрос на остановку отправлен. Обучение остановится на следующей эпохе...")
                st.rerun()
            force_reset_training = st.button("🧹 Сбросить зависший статус", width='stretch')
            if force_reset_training:
                reset_status = get_default_status()
                save_status(reset_status)
                st.success("Зависший статус обучения сброшен.")
                time.sleep(1)
                st.rerun()
        else:
            start_training = st.button("🚀 Начать Обучение", width='stretch', type="primary")
            if start_training:
                if len(png_files) == 0:
                    st.error("Ошибка: Папка с датасетом пуста! Сначала добавьте .png скины во вкладке 'Управление Датасетом'.")
                elif len(png_files) < batch_size_input:
                    st.warning(f"Внимание: количество скинов ({len(png_files)}) меньше размера батча ({batch_size_input}). Уменьшите размер батча.")
                else:
                    # Start thread!
                    t = threading.Thread(
                        target=run_training_thread,
                        args=(
                            data_dir_input,
                            DEFAULT_CHECKPOINT_DIR,
                            DEFAULT_SAMPLES_DIR,
                            epochs_input,
                            batch_size_input,
                            g_lr_input,
                            d_lr_input,
                            n_critic_input,
                            gp_lambda_input,
                            resume_input,
                            device
                        ),
                        daemon=True
                    )
                    t.start()
                    st.success("Обучение успешно запущено в фоновом режиме!")
                    time.sleep(1)
                    st.rerun()
                    
            # Clear logs option
            if training_status["status"] in ["completed", "error", "idle"] and len(training_status["loss_history"]["Epoch"]) > 0:
                if st.button("🗑️ Сбросить логи обучения", width='stretch'):
                    reset_status = get_default_status()
                    save_status(reset_status)
                    st.success("Логи успешно сброшены!")
                    time.sleep(1)
                    st.rerun()
        
    with col2:
        st.subheader("Прогресс обучения")
        
        # Load latest status inside col2 too to be fresh
        status = load_status()
        
        if status["status"] == "training":
            # Display real-time progress
            curr_epoch = status["current_epoch"]
            total_epochs = status["total_epochs"]
            d_loss = status["d_loss"]
            g_loss = status["g_loss"]
            critic_real = status["critic_real"]
            critic_fake = status["critic_fake"]
            gradient_penalty = status["gradient_penalty"]
            generator_updates = status["generator_updates"]
            
            # Progress bar
            progress_val = min(100, int((curr_epoch / max(1, total_epochs)) * 100))
            st.progress(progress_val)
            
            st.markdown(f"**Эпоха {curr_epoch}/{total_epochs}** | Текущие потери: *D Loss*: `{d_loss:.4f}`, *G Loss*: `{g_loss:.4f}`")
            st.markdown(f"**Диагностика критика:** *D(real)*: `{critic_real:.4f}`, *D(fake)*: `{critic_fake:.4f}`, *GP*: `{gradient_penalty:.4f}`, *G updates/epoch*: `{generator_updates}`")
            st.write("📈 *График потерь (обновляется в реальном времени):*")
            # Line chart
            if len(status["loss_history"]["Epoch"]) > 0:
                df_losses = pd.DataFrame(status["loss_history"]).set_index("Epoch")
                st.line_chart(df_losses)
                
            if len(status["diagnostics_history"]["Epoch"]) > 0:
                df_diagnostics = pd.DataFrame(status["diagnostics_history"]).set_index("Epoch")
                st.write("🔎 *Диагностика WGAN-GP:*")
                st.line_chart(df_diagnostics[["Critic Real", "Critic Fake", "Gradient Penalty"]])
                
            # Preview of current epoch samples
            if curr_epoch > 0:
                preview_epoch = find_latest_sample_epoch(DEFAULT_SAMPLES_DIR, max_epoch=curr_epoch)

                preview_files = []
                if preview_epoch is not None:
                    preview_files = [f for f in os.listdir(DEFAULT_SAMPLES_DIR) if f.startswith(f"epoch_{preview_epoch}_")]
                if len(preview_files) > 0:
                    st.write(f"**Промежуточный результат генерации нейросети (эпоха {preview_epoch}):**")
                    cols = st.columns(min(len(preview_files), 4))
                    for idx, p_file in enumerate(preview_files[:4]):
                        p_img = Image.open(os.path.join(DEFAULT_SAMPLES_DIR, p_file))
                        p_label = p_file.split("epoch_")[-1].replace(".png", "").replace("_", " ")
                        if len(p_label) > 25:
                            p_label = p_label[:22] + "..."
                        cols[idx].image(p_img.resize((128, 128), Image.Resampling.NEAREST), caption=p_label, width='stretch')
            
            # Sleep and rerun to auto-refresh
            time.sleep(2.5)
            st.rerun()
            
        elif status["status"] == "completed":
            st.success("🎉 Обучение успешно завершено! Модель готова к генерации во вкладке 'Генерация Скинов'.")
            
            # Show final stats and final chart
            if len(status["loss_history"]["Epoch"]) > 0:
                df_losses = pd.DataFrame(status["loss_history"]).set_index("Epoch")
                st.line_chart(df_losses)
                
            # Show samples from the final epoch
            final_epoch = status["current_epoch"]
            preview_files = [f for f in os.listdir(DEFAULT_SAMPLES_DIR) if f.startswith(f"epoch_{final_epoch}_")]
            if len(preview_files) > 0:
                st.write("**Итоговые результаты генерации (samples):**")
                cols = st.columns(min(len(preview_files), 4))
                for idx, p_file in enumerate(preview_files[:4]):
                    p_img = Image.open(os.path.join(DEFAULT_SAMPLES_DIR, p_file))
                    p_label = p_file.split("epoch_")[-1].replace(".png", "").replace("_", " ")
                    cols[idx].image(p_img.resize((128, 128), Image.Resampling.NEAREST), caption=p_label, width='stretch')
                    
        elif status["status"] == "error":
            st.error("❌ Произошла ошибка при обучении:")
            st.code(status["error_message"])
            
            # Show chart of whatever was completed
            if len(status["loss_history"]["Epoch"]) > 0:
                df_losses = pd.DataFrame(status["loss_history"]).set_index("Epoch")
                st.line_chart(df_losses)
                
        else:
            # idle status
            # If there's previous history, show it!
            if len(status["loss_history"]["Epoch"]) > 0:
                st.info("Обучение остановлено/не запущено. Ниже показан график предыдущей сессии:")
                df_losses = pd.DataFrame(status["loss_history"]).set_index("Epoch")
                st.line_chart(df_losses)
            else:
                st.info("Здесь будет отображаться лосс-график, прогресс-бар и тестовые генерации во время обучения нейросети.")

# ==========================================
# TAB 3: DATASET MANAGEMENT
# ==========================================
with tabs[2]:
    st.header("📁 Управление Датасетом")
    
    st.write("""
    Чтобы нейросеть научилась рисовать скины, ей нужен обучающий материал.
    
    ### Как подготовить датасет:
    1. Загрузите скины в формате `.png` (размер 64x64 или 64x32).
    2. Опишите каждый скин. Для этого есть 3 способа:
       - **Способ А (Рекомендуемый):** Назовите файл понятным именем. Например, `dark_knight_red_crown.png`. Нейросеть сама очистит имя и поймет, что это "dark knight red crown" (темный рыцарь красная корона).
       - **Способ Б:** Создайте текстовый файл `.txt` с таким же именем. Например, к файлу `skin_1.png` приложите `skin_1.txt`, в котором напишите описание, например: `рыцарь в синей броне`.
       - **Способ В:** Создайте один файл `metadata.json` в папке датасета, в котором сопоставьте имена файлов с описаниями:
         ```json
         {
           "skin1.png": "световой эльф с луком",
           "skin2.png": "киберпанк ниндзя с маской"
         }
         ```
    """)
    
    st.divider()
    
    # Procedural generation option
    st.subheader("🎁 Сгенерировать готовый демо-датасет")
    st.write("""
    Если у вас нет готовых скинов, вы можете мгновенно сгенерировать **100 красивых процедурных скинов** 
    разных стилей (огненные демоны, ледяные маги, эльфы, кибер-ниндзя, космонавты) с описаниями сразу на русском и английском языках!
    Это позволит вам запустить обучение и проверить работу нейросети прямо сейчас.
    """)
    if st.button("⚙️ Сгенерировать 100 демо-скинов", width='stretch'):
        with st.spinner("Создание красивых текстур скинов..."):
            generate_test_dataset(data_dir=DEFAULT_DATA_DIR, count_per_theme=10)
            st.success("Успешно создано 100 скинов с текстовыми описаниями в папке 'dataset'!")
            time.sleep(1)
            st.rerun()

    st.divider()
    
    # Upload interface
    st.subheader("📥 Быстрая загрузка обучающих материалов")
    
    uploaded_files = st.file_uploader(
        "Перетащите .png скины сюда:", 
        type=["png"], 
        accept_multiple_files=True,
        help="Загруженные файлы будут сохранены в папку 'dataset'"
    )
    
    if uploaded_files:
        save_count = 0
        for uploaded_file in uploaded_files:
            file_path = os.path.join(DEFAULT_DATA_DIR, uploaded_file.name)
            # Save PNG skin
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            save_count += 1
            
        st.success(f"Успешно сохранено скинов в папку '{DEFAULT_DATA_DIR}': **{save_count} шт.**")
        st.rerun()

    # View dataset
    st.subheader("🖼️ Обзор вашего датасета")
    
    all_files = [f for f in os.listdir(DEFAULT_DATA_DIR) if f.lower().endswith(".png")]
    
    if len(all_files) == 0:
        st.info("Ваш датасет пуст. Пожалуйста, загрузите скины выше, чтобы начать обучение.")
    else:
        # Load dataset items list
        dataset_items = []
        # Temporary load text encoder to parse prompts cleanly
        temp_dataset = MinecraftSkinDataset(data_dir=DEFAULT_DATA_DIR)
        
        for sample in temp_dataset.samples:
            dataset_items.append({
                "Имя файла": os.path.basename(sample["image_path"]),
                "Описание (Промпт)": sample["prompt"]
            })
            
        df_items = pd.DataFrame(dataset_items)
        st.dataframe(df_items, width='stretch')
        
        # Grid visualizer
        st.write("**Предпросмотр скинов датасета:**")
        cols = st.columns(6)
        for idx, sample in enumerate(temp_dataset.samples[:24]):
            col_idx = idx % 6
            img = Image.open(sample["image_path"])
            # Clearer view with Nearest Neighbor interpolation
            cols[col_idx].image(img.resize((96, 96), Image.Resampling.NEAREST), caption=os.path.basename(sample["image_path"]), width='stretch')
            
        if len(temp_dataset.samples) > 24:
            st.write(f"*И еще {len(temp_dataset.samples) - 24} скинов в папке датасета...*")
