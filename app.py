import os
import pandas as pd
import numpy as np
from flask import Flask, render_template, request
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'data_file' not in request.files:
            return render_template('index.html', error='Файл не загружен')
        file = request.files['data_file']
        if file.filename == '':
            return render_template('index.html', error='Файл не выбран')
        if not file.filename.endswith('.csv'):
            return render_template('index.html', error='Разрешены только CSV файлы')

        try:
            df = pd.read_csv(file, encoding='utf-8')
            required_cols = ['user_id', 'item_id', 'rating']
            missing = [c for c in required_cols if c not in df.columns]
            if missing:
                return render_template('index.html', error=f'В CSV отсутствуют столбцы: {", ".join(missing)}')
            
            df['rating'] = pd.to_numeric(df['rating'], errors='coerce').fillna(0)
            df.drop_duplicates(subset=['user_id', 'item_id'], inplace=True)
            
            if len(df['item_id'].unique()) < 2:
                return render_template('index.html', error='В датасете должно быть минимум 2 уникальных товара')
                
        except Exception as e:
            return render_template('index.html', error=f'Ошибка обработки данных: {str(e)}')

        try:
            current_conversion = float(request.form.get('conversion', 2.0))
            avg_order_value = float(request.form.get('aov', 3000))
            monthly_visitors = int(request.form.get('visitors', 10000))
            ai_cost = float(request.form.get('ai_cost', 500000))

            user_item_matrix = df.pivot_table(index='user_id', columns='item_id', values='rating', fill_value=0)
            item_similarity = cosine_similarity(user_item_matrix.T)

            target_idx = 0
            similarity_scores = list(enumerate(item_similarity[target_idx]))
            similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[1:6]
            
            recommended_items = [user_item_matrix.columns[idx] for idx, _ in similarity_scores]
            similarity_values = [round(float(score), 3) for _, score in similarity_scores]
            
            # Дополняем до 5 элементов для стабильности шаблона
            while len(recommended_items) < 5:
                recommended_items.append("—")
                similarity_values.append("—")

           # Задание эмпирических диапазонов роста конверсии (10-30%) на основе отраслевых данных (Табл. 2.3 диплома)
            conv_uplift_min, conv_uplift_max = 0.10, 0.30
            # Задание диапазонов роста среднего чека (5-20%) от персонализированных кросс-продаж (Гл. 3.2, 3.4)
            aov_uplift_min, aov_uplift_max = 0.05, 0.20
            # Расчёт текущей ежемесячной выручки по формуле: посетители × конверсия × средний чек
            current_monthly_revenue = monthly_visitors * (current_conversion / 100) * avg_order_value
            # Расчёт минимальной прогнозной выручки с учётом консервативного uplift-а конверсии и AOV
            projected_revenue_min = monthly_visitors * ((current_conversion * (1 + conv_uplift_min)) / 100) * avg_order_value * (1 + aov_uplift_min)
            # Расчёт максимальной прогнозной выручки по оптимистичному сценарию внедрения ИИ
            projected_revenue_max = monthly_visitors * ((current_conversion * (1 + conv_uplift_max)) / 100) * avg_order_value * (1 + aov_uplift_max)
            # Определение абсолютного прироста выручки при минимальном сценарии
            revenue_increase_min = projected_revenue_min - current_monthly_revenue
            # Определение абсолютного прироста выручки при максимальном сценарии
            revenue_increase_max = projected_revenue_max - current_monthly_revenue
            # Прогнозирование годовой финансовой выгоды на основе месячного прироста (минимальный сценарий)
            annual_benefit_min = revenue_increase_min * 12
            # Прогнозирование максимальной годовой выгоды от ИИ-персонализации
            annual_benefit_max = revenue_increase_max * 12
            # Расчёт минимального ROI по формуле из Гл. 2.2: (Выгоды − Затраты) / Затраты × 100%
            roi_min = round(((annual_benefit_min - ai_cost) / ai_cost) * 100, 2)
            # Расчёт максимального ROI для демонстрации диапазона окупаемости проекта
            roi_max = round(((annual_benefit_max - ai_cost) / ai_cost) * 100, 2)

            fig, ax = plt.subplots(figsize=(6, 4))
            categories = ['Текущая выручка', 'Прогноз (мин)', 'Прогноз (макс)']
            values = [current_monthly_revenue, projected_revenue_min, projected_revenue_max]
            ax.bar(categories, values, color=['#4a90e2', '#50c878', '#f5a623'])
            ax.set_ylabel('Выручка (руб)')
            ax.set_title('Прогноз экономической эффективности')
            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            chart_base64 = base64.b64encode(buf.read()).decode('utf-8')
            plt.close(fig)

            return render_template('index.html',
                                   recommended_items=recommended_items,
                                   similarity_values=similarity_values,
                                   current_revenue=current_monthly_revenue,
                                   projected_revenue_min=projected_revenue_min,
                                   projected_revenue_max=projected_revenue_max,
                                   roi_min=roi_min,
                                   roi_max=roi_max,
                                   chart_image=chart_base64,
                                   error=None)
        except Exception as e:
            return render_template('index.html', error=f'Ошибка расчётов: {str(e)}')
            
    return render_template('index.html', error=None)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
