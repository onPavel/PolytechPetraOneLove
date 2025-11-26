
# Пример Python скрипта для анализа рынка MENA для бренда BIOMED
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import requests
from datetime import datetime

class MENAMarketAnalyzer:
    """Класс для анализа рынка MENA для косметических брендов"""

    def __init__(self):
        self.market_data = None
        self.competitor_data = None
        self.customer_segments = None

    def load_market_data(self, file_path):
        """Загрузка данных о рынке MENA"""
        try:
            self.market_data = pd.read_csv(file_path, encoding='utf-8')
            print(f"Данные загружены: {len(self.market_data)} стран")
            return self.market_data
        except FileNotFoundError:
            print("Файл с данными не найден")
            return None

    def calculate_market_potential(self):
        """Расчет потенциала рынка для каждой страны"""
        if self.market_data is None:
            print("Сначала загрузите данные")
            return None

        # Расчет индекса привлекательности рынка
        self.market_data['Market_Potential'] = (
            self.market_data['Косметический_рынок_млн_USD'] * 0.4 +
            self.market_data['ВВП_на_душу_тыс_USD'] * 10 * 0.3 +
            self.market_data['Проникновение_интернета_%'] * 5 * 0.2 +
            self.market_data['Рост_eCommerce_%'] * 10 * 0.1
        )

        return self.market_data.sort_values('Market_Potential', ascending=False)

    def segment_countries(self, n_clusters=3):
        """Сегментация стран с помощью кластеризации"""
        if self.market_data is None:
            print("Сначала загрузите данные")
            return None

        # Подготовка данных для кластеризации
        features = ['ВВП_на_душу_тыс_USD', 'Проникновение_интернета_%', 
                   'Косметический_рынок_млн_USD', 'Рост_eCommerce_%']

        X = self.market_data[features]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Кластеризация
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.market_data['Segment'] = kmeans.fit_predict(X_scaled)

        # Определение характеристик сегментов
        segment_names = {0: 'Премиум рынки', 1: 'Развивающиеся рынки', 2: 'Массовые рынки'}
        self.market_data['Segment_Name'] = self.market_data['Segment'].map(segment_names)

        return self.market_data

    def generate_pricing_strategy(self, country):
        """Генерация стратегии ценообразования для конкретной страны"""
        if self.market_data is None:
            print("Сначала загрузите данные")
            return None

        country_data = self.market_data[self.market_data['Страна'] == country]

        if country_data.empty:
            print(f"Данные для страны {country} не найдены")
            return None

        gdp_per_capita = country_data['ВВП_на_душу_тыс_USD'].iloc[0]
        market_size = country_data['Косметический_рынок_млн_USD'].iloc[0]

        # Базовая цена продукта (например, крем для лица)
        base_price_usd = 25

        if gdp_per_capita > 30:
            price_multiplier = 1.2  # Премиум ценообразование
            strategy = "Премиум позиционирование"
        elif gdp_per_capita > 15:
            price_multiplier = 1.0  # Стандартное ценообразование
            strategy = "Средний сегмент"
        else:
            price_multiplier = 0.8  # Доступное ценообразование
            strategy = "Массовый сегмент"

        recommended_price = base_price_usd * price_multiplier

        return {
            'country': country,
            'recommended_price_usd': recommended_price,
            'strategy': strategy,
            'market_size_million': market_size,
            'gdp_per_capita': gdp_per_capita
        }

    def create_market_dashboard_data(self):
        """Создание данных для дашборда"""
        if self.market_data is None:
            print("Сначала загрузите данные")
            return None

        dashboard_data = {
            'top_markets': self.market_data.nlargest(5, 'Косметический_рынок_млн_USD'),
            'high_growth_markets': self.market_data.nlargest(5, 'Рост_eCommerce_%'),
            'premium_markets': self.market_data[self.market_data['ВВП_на_душу_тыс_USD'] > 20],
            'total_market_size': self.market_data['Косметический_рынок_млн_USD'].sum(),
            'total_population': self.market_data['Население_млн'].sum()
        }

        return dashboard_data

# Пример использования
if __name__ == "__main__":
    # Инициализация анализатора
    analyzer = MENAMarketAnalyzer()

    # Загрузка данных
    market_data = analyzer.load_market_data('mena_market_analysis.csv')

    # Расчет потенциала рынков
    potential_analysis = analyzer.calculate_market_potential()
    print("\nТоп-5 рынков по потенциалу:")
    print(potential_analysis[['Страна', 'Market_Potential']].head())

    # Сегментация стран
    segmented_data = analyzer.segment_countries()
    print("\nСегментация стран:")
    for segment in segmented_data['Segment_Name'].unique():
        countries = segmented_data[segmented_data['Segment_Name'] == segment]['Страна'].tolist()
        print(f"{segment}: {', '.join(countries)}")

    # Стратегия ценообразования для ОАЭ
    pricing_uae = analyzer.generate_pricing_strategy('ОАЭ')
    print(f"\nСтратегия ценообразования для ОАЭ:")
    print(f"Рекомендуемая цена: ${pricing_uae['recommended_price_usd']:.2f}")
    print(f"Стратегия: {pricing_uae['strategy']}")

    # Создание данных для дашборда
    dashboard_data = analyzer.create_market_dashboard_data()
    print(f"\nОбщий размер рынка MENA: ${dashboard_data['total_market_size']:,.0f} млн")
    print(f"Общее население: {dashboard_data['total_population']:.1f} млн")
