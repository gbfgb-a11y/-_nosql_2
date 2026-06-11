1.2:
    Pinecone це «хмара під ключ»: нічого налаштовувати не треба, але платно і закрито. Підійде для бізнес-проєкту, де гроші є, а свого сисадміна нема.

    Qdrant відкритий, потужний, ставиш собі на сервер. Безкоштовно, але треба вміти налаштовувати. Для команди, яка хоче контролювати все сама.

    Chroma легкий, як блокнот. Ідеально для експериментів удома чи в ноутбуці. Для великих даних не піде.

    Коротко: Pinecone як таксі (зручно, але платиш). Qdrant власне авто (треба вміти водити). Chroma — велосипед для прогулянок.

1.3: 
    Косинусна схожість це наче кут між двома векторами. Якщо вони спрямовані в один бік кут нуль, косинус дорівнює 1 значить, тексти дуже схожі. Формула там така: скалярний добуток поділити на довжини векторів.

    Але якщо заздалегідь зробити всі вектори одиничної довжини (тобто "нормалізувати" їх), то довжина кожного стає 1. А ділення на 1 нічого не змінює. Тому залишається просто скалярний добуток.

3:
    1:
        Чи збігаються топ-5 для cosine і dot product і чому?:
        Так, у проведеному експерименті топ-5 результатів для cosine similarity і dot product повністю збіглися. Це пояснюється тим, що під час створення ембеддингів використовувалась нормалізація (normalize_embeddings=True), тому всі вектори мають одиничну довжину. Для нормалізованих векторів косинусна схожість і скалярний добуток є математично еквівалентними, тому вони формують однакове ранжування документів. 

    2:
        Чи відрізняються результати для L2 і чому?:
        У проведеному експерименті результати для L2-distance також збіглися за порядком документів із cosine similarity та dot product. Проте числові значення метрики були іншими, оскільки L2 вимірює відстань між векторами, а cosine similarity кутову схожість. Для нормалізованих векторів існує прямий зв’язок між цими величинами, тому порядок ранжування залишається однаковим.

    3:
        Що сталося б, якби ембеддинги не були нормалізовані?:
        Без нормалізації довжина векторів почала б впливати на результат пошуку. У такому випадку cosine similarity враховувала б лише напрямок векторів, тоді як dot product додатково залежав би від їхньої довжини. Через це результати ранжування могли б суттєво відрізнятися. Документи з більшими за модулем векторами отримували б перевагу навіть за меншої семантичної схожості.

4:
    1:
        Більш осмислені результати показало семантичне розбиття (Semantic Chunking). Цей підхід об’єднує завершені речення в один фрагмент тексту, завдяки чому зберігається логічна структура документа. У результаті ембеддинги краще відображають зміст тексту та забезпечують якісніший пошук.

    2:
        При використанні Fixed-size chunking окремі речення можуть бути розділені між двома чанками. У таких випадках частина контексту втрачається, а ембеддинг містить неповну інформацію про зміст речення. Це може негативно впливати на релевантність результатів пошуку. Semantic Chunking практично усуває цю проблему завдяки збереженню меж речень.

    3:
        Зі збільшенням overlap кількість чанків зростає, оскільки частина тексту дублюється між сусідніми фрагментами. Це покращує збереження контексту та зменшує втрати інформації на межах чанків. Проте збільшення overlap також підвищує обсяг збережених даних і кількість ембеддингів, що потребує більше пам’яті та ресурсів для індексації.

5:
    1:
        Гібридний пошук (BM25 + вектори + RRF) дав кращий результат, ніж кожен з методів окремо. Чому? Тому що BM25 добре знаходить точні терміни (наприклад, «BERT fine-tuning»), але не розуміє синоніми та перефразування. Векторний пошук, навпаки, знаходить статті за змістом («masked language model pre-training»), але може пропустити рідкісний, але важливий термін. Гібрид об'єднує сили обох: якщо документ релевантний, він високо підніметься хоча б в одному зі списків, і RRF це підсилює. У підсумку гібрид видає більш релевантні та різноманітні результати.

    2:
        Так, таке часто трапляється. Уявімо, що якийсь документ займає 6-те місце у векторному пошуку і 7-ме у BM25. Окремо він не потрапляє в топ-5 жодного з методів. Але RRF підсумовує зворотні ранги: для 6-ї позиції це 1/(60+5)=1/65, для 7-ї — 1/(60+6)=1/66. Разом це дає суму, яка може випередити документи, що добре виступили лише в одному з методів, але погано в іншому. Отже, гібридний пошук може «витягнути» компромісний документ, який обидва методи вважають «досить добрим», — і він опиняється в топ-5 гібрида, хоча його не було в топ-5 жодного окремого методу.

    3:
        Параметр k в RRF (в формулі 1/(k + rank)) контролює, наскільки сильно високі ранги домінують над низькими.
        k = 1 різниця між 1-м і 2-м місцем величезна (1/2 vs 1/3 ≈ 0.5 vs 0.33). Тобто перші місця отримують дуже велику вагу. Це добре, якщо ви впевнені, що топ-1 кожного методу ідеальний. Але якщо один метод помилиться і поставить нерелевантний документ на перше місце, він зіпсує видачу.
        k = 60 — всі ранги отримують менші, але більш згладжені ваги (1/61 vs 1/62 ≈ 0.0164 vs 0.0161). Різниця майже непомітна. Таким чином, результати стабільніші, але перші місця не так домінують.
        На практиці беруть k=60 (класичне значення з оригінальної статті про RRF). Це дає збалансоване поєднання: метод не надто чутливий до помилок ранжування, але все ж надає перевагу вищим позиціям. Якщо взяти k=1, видача буде сильно залежати від топ-1 кожного методу — може стати гіршою, якщо один з методів десь схибив.

6:
    1:
        У ході роботи було протестовано BM25, векторний пошук та їх поєднання.
        Для запиту "BERT fine-tuning" BM25 показав слабкі результати, оскільки в датасеті переважно містилися статті 2007–2009 років, коли модель BERT ще не існувала. Векторний пошук також не зміг знайти релевантні документи через відсутність відповідної тематики в колекції. Для запиту "making computers understand human emotions from text" векторний пошук показав кращі результати, оскільки зміг знайти статті, пов’язані з емоціями та афективними обчисленнями, навіть без точного збігу ключових слів. BM25 також знайшов релевантні документи, проте значною мірою спирався на прямий збіг термінів.
        Для запиту "Yann LeCun convolutional networks" BM25 знаходив документи зі словом convolutional, але не завжди пов’язані з нейронними мережами. Векторний пошук повертав результати, пов’язані з машинним навчанням та класифікацією, тобто краще враховував семантичний зміст запиту. Загальне правило полягає в тому, що BM25 доцільно використовувати для точних запитів, які містять назви моделей, терміни, абревіатури або імена авторів. Векторний пошук краще працює для природномовних запитів, перефразувань і пошуку за змістом, коли користувач описує задачу своїми словами.

    2:
        Якщо чанк занадто малий (10–15 слів), він містить недостатньо контексту для побудови якісного ембеддингу. У результаті пошукова система може втрачати важливі зв’язки між поняттями, а релевантність результатів знижується.
        Якщо чанк занадто великий (500 і більше слів), у ньому можуть одночасно міститися кілька різних тем. Ембеддинг такого фрагмента стає усередненим представленням великого обсягу інформації, що також може погіршувати якість пошуку.
        Оптимальний розмір чанка залежить від конкретної задачі та структури документів. Для наукових текстів зазвичай використовують фрагменти приблизно від 100 до 300 слів, оскільки вони забезпечують баланс між збереженням контексту та точністю пошуку.

    3:
        У роботі використовувалися нормалізовані ембеддинги (normalize_embeddings=True), тобто всі вектори мають одиничну довжину:
        ∣∣x∣∣=∣∣y∣∣=1
        Для таких векторів евклідова відстань обчислюється як:
        ∣∣x−y∣∣**2 = (x−y) * (x−y)
        Після розкриття дужок:
        ∣∣x−y∣∣**2 = x * x + y * y − 2 x * y
        Оскільки вектори нормалізовані:
        x⋅x=1, y⋅y=1
        отримуємо:
        ∣∣x−y∣∣**2 = 2 − 2cos(θ)
        Таким чином, для одиничних векторів мінімізація евклідової відстані еквівалентна максимізації косинусної схожості. Тому, якби індекс Pinecone був створений з метрикою euclidean, результати пошуку практично не відрізнялися б від використання cosine similarity.

    4:
        Під час виконання роботи використовувався безкоштовний тариф Pinecone Starter. Основними обмеженнями такого тарифу є обмежена кількість індексів, обмежений обсяг збережених векторів, нижча продуктивність та обмежені ресурси для масштабування.
        Для датасету з 10 000 статей цих можливостей достатньо, проте для колекції з 10 мільйонів документів виникли б серйозні проблеми зі зберіганням, індексацією та швидкістю пошуку. У випадку роботи з 10 мільйонами статей доцільно використовувати платний кластер Pinecone або іншу масштабовану векторну базу даних, застосовувати шардінг індексів, пакетне завантаження даних та багаторівневий пошук. Також можна використовувати гібридну схему, де BM25 виконує попередню фільтрацію великої колекції, а векторний пошук застосовується лише до обмеженого набору кандидатів. Це дозволяє значно зменшити витрати ресурсів та прискорити пошук.


Результати:
    01:
        Збережено 10000 записів у data\arxiv_subset.parquet

    02:
        No sentence-transformers model found with name allenai/specter2_base. Creating a new one with mean pooling.
        Batches: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 313/313 [46:55<00:00,  9.00s/it]
        Збережено (10000, 768) вектори в embeddings/embeddings.npy

    03:
        ...
        Завантажено 9900 з 10000 векторів
        Завантажено 10000 з 10000 векторів
        Всі вектори завантажено в Pinecone
        Статистика індексу: {'dimension': 768,
        'index_fullness': 0.0,
        'namespaces': {'': {'vector_count': 10000}},
        'total_vector_count': 10000}

    04:
        ============================================================
        Семантичний пошук: 'teaching machines to recognize objects in pictures'
        ============================================================

        1. Learning View Generalization Functions
        Категорія: cs.CV  |  Рік: 2007.0
        Score: 0.5350
        Abstract: Learning object models from views in 3D visual object recognition is usually
        formulated either as a function approximati...

        2. Visual object categorization with new keypoint-based adaBoost features
        Категорія: cs.CV  |  Рік: 2009.0
        Score: 0.4812
        Abstract: We present promising results for visual object categorization, obtained with
        adaBoost using new original ?keypoints-base...

        3. Learning Object Location Predictors with Boosting and Grammar-Guided
        Feature Extraction
        Категорія: cs.CV  |  Рік: 2009.0
        Score: 0.4731
        Abstract: We present BEAMER: a new spatially exploitative approach to learning object
        detectors which shows excellent results when...

        4. Human expert fusion for image classification
        Категорія: cs.CV  |  Рік: 2008.0
        Score: 0.4652
        Abstract: In image classification, merging the opinion of several human experts is very
        important for different tasks such as the ...

        5. Adaboost with "Keypoint Presence Features" for Real-Time Vehicle Visual
        Detection
        Категорія: cs.CV  |  Рік: 2009.0
        Score: 0.4504
        Abstract: We present promising results for real-time vehicle visual detection, obtained
        with adaBoost using new original ?keypoint...

        ============================================================
        Фільтр A: RL після 2000, cs.LG
        ============================================================

        1. Feature Reinforcement Learning: Part I: Unstructured MDPs
        Категорія: cs.LG  |  Рік: 2009.0
        Score: 0.5623
        Abstract: General-purpose, intelligent, learning agents cycle through sequences of
        observations, actions, and rewards that are com...

        2. Temporal Difference Updating without a Learning Rate
        Категорія: cs.LG  |  Рік: 2008.0
        Score: 0.5478
        Abstract: We derive an equation for temporal difference learning from statistical
        principles. Specifically, we start with the vari...

        3. Rollout Sampling Approximate Policy Iteration
        Категорія: cs.LG  |  Рік: 2008.0
        Score: 0.5424
        Abstract: Several researchers have recently investigated the connection between
        reinforcement learning and classification. We are ...

        4. On the Possibility of Learning in Reactive Environments with Arbitrary
        Dependence
        Категорія: cs.LG  |  Рік: 2008.0
        Score: 0.5308
        Abstract: We address the problem of reinforcement learning in which observations may
        exhibit an arbitrary form of stochastic depen...

        5. A Convergent Online Single Time Scale Actor Critic Algorithm
        Категорія: cs.LG  |  Рік: 2009.0
        Score: 0.5175
        Abstract: Actor-Critic based approaches were among the first to address reinforcement
        learning in a general setting. Recently, the...

        ============================================================
        Фільтр B: до 2010, будь-яка категорія
        ============================================================

        1. Reinforcement Learning by Value Gradients
        Категорія: cs.NE  |  Рік: 2008.0
        Score: 0.5693
        Abstract: The concept of the value-gradient is introduced and developed in the context
        of reinforcement learning. It is shown that...

        2. A Monte Carlo AIXI Approximation
        Категорія: cs.AI  |  Рік: 2009.0
        Score: 0.5664
        Abstract: This paper introduces a principled approach for the design of a scalable
        general reinforcement learning agent. Our appro...

        3. Feature Reinforcement Learning: Part I: Unstructured MDPs
        Категорія: cs.LG  |  Рік: 2009.0
        Score: 0.5623
        Abstract: General-purpose, intelligent, learning agents cycle through sequences of
        observations, actions, and rewards that are com...

        4. Time manipulation technique for speeding up reinforcement learning in
        simulations
        Категорія: cs.AI  |  Рік: 2009.0
        Score: 0.5542
        Abstract: A technique for speeding up reinforcement learning algorithms by using time
        manipulation is proposed. It is applicable t...

        5. Temporal Difference Updating without a Learning Rate
        Категорія: cs.LG  |  Рік: 2008.0
        Score: 0.5478
        Abstract: We derive an equation for temporal difference learning from statistical
        principles. Specifically, we start with the vari...

        ============================================================
        Порівняння метрик для: 'attention mechanism in neural networks'
        ============================================================

        --- Cosine similarity ---
        1. [0.5832] A computational approach to the covert and overt deployment 
        2. [0.5084] Cognitive Architecture for Direction of Attention Founded on
        3. [0.4310] Computer Model of a "Sense of Humour". II. Realization in Ne
        4. [0.3991] Another Look at Quantum Neural Computing
        5. [0.3914] Brain architecture: A design for natural computation

        --- Dot product ---
        1. [0.5832] A computational approach to the covert and overt deployment 
        2. [0.5084] Cognitive Architecture for Direction of Attention Founded on
        3. [0.4310] Computer Model of a "Sense of Humour". II. Realization in Ne
        4. [0.3991] Another Look at Quantum Neural Computing
        5. [0.3914] Brain architecture: A design for natural computation

        --- L2 distance ---
        1. [0.9131] A computational approach to the covert and overt deployment 
        2. [0.9916] Cognitive Architecture for Direction of Attention Founded on
        3. [1.0667] Computer Model of a "Sense of Humour". II. Realization in Ne
        4. [1.0962] Another Look at Quantum Neural Computing
        5. [1.1032] Brain architecture: A design for natural computation

    05: 
        ============================================================
        Запит: neural network training optimization
        ============================================================

        -- Fixed chunking --
        [0.3479] Node harvest
                chunk: new observation falls typically into several nodes and its prediction is then the weighted average o...
        [0.3075] Node harvest
                chunk: to reconcile the two aims of interpretability and predictive accuracy by combining positive aspects ...
        [0.3010] Passive network tomography for erroneous networks:
                chunk: coding properties of RLNC. In particular, we design network codes based on Reed-Solomon codes so tha...
        [0.2590] Pac-Bayesian Supervised Classification: The Thermo
                chunk: Vapnik's generalization bounds, extending them to the case when the sample is made of independent no...
        [0.2554] Pac-Bayesian Supervised Classification: The Thermo
                chunk: whose expected error rate converges according to the best possible power of the sample size adaptive...

        -- Semantic chunking --
        [0.3947] Node harvest
                chunk: Results are very sparse
        and interpretable and predictive accuracy is extremely competitive, especial...
        [0.3234] Node harvest
                chunk: The only role of node harvest is to
        `pick' the right nodes from the initial large ensemble of nodes ...
        [0.2916] Passive network tomography for erroneous networks:
                chunk: In particular, we design network
        codes based on Reed-Solomon codes so that a maximal number of adver...
        [0.2801] Passive network tomography for erroneous networks:
                chunk: Our algorithm for topology estimation with random network errors has
        time complexity that is polynom...
        [0.2754] Pac-Bayesian Supervised Classification: The Thermo
                chunk: Finally we review briefly the construction of Support
        Vector Machines and show how to derive general...

        ============================================================
        Запит: natural language processing text classification
        ============================================================

        -- Fixed chunking --
        [0.3532] DBMSs Should Talk Back Too
                chunk: DBMSs Should Talk Back Too [SEP] Natural language user interfaces to database systems have been stud...
        [0.3328] Pac-Bayesian Supervised Classification: The Thermo
                chunk: Vapnik's generalization bounds, extending them to the case when the sample is made of independent no...
        [0.3061] DBMSs Should Talk Back Too
                chunk: addressed. In this paper, we first expose the reader to several situations and applications that nee...
        [0.3042] DBMSs Should Talk Back Too
                chunk: information extraction has received considerable attention in the past ten years or so, identifying ...
        [0.2802] Pac-Bayesian Supervised Classification: The Thermo
                chunk: Pac-Bayesian Supervised Classification: The Thermodynamics of Statistical Learning [SEP] This monogr...

        -- Semantic chunking --
        [0.3749] Pac-Bayesian Supervised Classification: The Thermo
                chunk: Finally we review briefly the construction of Support
        Vector Machines and show how to derive general...
        [0.3093] DBMSs Should Talk Back Too
                chunk: In this paper, we first expose
        the reader to several situations and applications that need translati...
        [0.2862] DBMSs Should Talk Back Too
                chunk: Likewise, information extraction has received considerable attention in the
        past ten years or so, id...
        [0.2848] Coding Theory and Projective Spaces
                chunk: We present efficient enumerative encoding and
        decoding techniques for the Grassmannian. Finally we d...
        [0.2728] Pac-Bayesian Supervised Classification: The Thermo
                chunk: We then
        discuss relative bounds, comparing the generalization error of two
        classification rules, sho...

    06:
        ============================================================
        Запит: BERT fine-tuning
        ============================================================

        --- BM25: 'BERT fine-tuning' ---
        [8.1136] Self-Improving Algorithms
        [0.0000] Context-free pairs of groups I: Context-free pairs and graph
        [0.0000] A Secure Communication Game with a Relay Helping the Eavesdr
        [0.0000] Sorting under Partial Information (without the Ellipsoid Alg
        [0.0000] Learning Exponential Families in High-Dimensions: Strong Con

        --- Векторний: 'BERT fine-tuning' ---
        [0.3553] Bounding the Probability of Error for High Precision Recogni
        [0.3362] PERCEVAL: a Computer-Driven System for Experimentation on Au
        [0.3284] Effect of Tuned Parameters on a LSA MCQ Answering Model
        [0.3232] Tests of Machine Intelligence
        [0.3207] A Better Good-Turing Estimator for Sequence Probabilities

        --- Гібридний RRF (k=60): 'BERT fine-tuning' ---
        [RRF=0.01639] Self-Improving Algorithms
        [RRF=0.01639] Bounding the Probability of Error for High Precision Recogni
        [RRF=0.01613] Context-free pairs of groups I: Context-free pairs and graph
        [RRF=0.01613] PERCEVAL: a Computer-Driven System for Experimentation on Au
        [RRF=0.01587] A Secure Communication Game with a Relay Helping the Eavesdr

        --- Гібридний RRF (k=1): 'BERT fine-tuning' ---
        [RRF=0.50000] Self-Improving Algorithms
        [RRF=0.50000] Bounding the Probability of Error for High Precision Recogni
        [RRF=0.33333] Context-free pairs of groups I: Context-free pairs and graph
        [RRF=0.33333] PERCEVAL: a Computer-Driven System for Experimentation on Au
        [RRF=0.25000] A Secure Communication Game with a Relay Helping the Eavesdr

        ============================================================
        Запит: Yann LeCun convolutional networks
        ============================================================

        --- BM25: 'Yann LeCun convolutional networks' ---
        [13.8360] Network error correction for unit-delay, memory-free network
        [12.3507] Convolutional Entanglement Distillation
        [12.1902] Entanglement-Assisted Quantum Convolutional Coding
        [11.5861] Convolutional codes from units in matrix and group rings
        [11.5405] Receding horizon decoding of convolutional codes

        --- Векторний: 'Yann LeCun convolutional networks' ---
        [0.4149] Large-Margin kNN Classification Using a Deep Encoder Network
        [0.3979] Minimum Probability Flow Learning
        [0.3883] Multi-Dimensional Recurrent Neural Networks
        [0.3776] Laplacian Support Vector Machines Trained in the Primal
        [0.3760] Human expert fusion for image classification

        --- Гібридний RRF (k=60): 'Yann LeCun convolutional networks' ---
        [RRF=0.01639] Network error correction for unit-delay, memory-free network
        [RRF=0.01639] Large-Margin kNN Classification Using a Deep Encoder Network
        [RRF=0.01613] Convolutional Entanglement Distillation
        [RRF=0.01613] Minimum Probability Flow Learning
        [RRF=0.01587] Entanglement-Assisted Quantum Convolutional Coding

        --- Гібридний RRF (k=1): 'Yann LeCun convolutional networks' ---
        [RRF=0.50000] Network error correction for unit-delay, memory-free network
        [RRF=0.50000] Large-Margin kNN Classification Using a Deep Encoder Network
        [RRF=0.33333] Convolutional Entanglement Distillation
        [RRF=0.33333] Minimum Probability Flow Learning
        [RRF=0.25000] Entanglement-Assisted Quantum Convolutional Coding

        ============================================================
        Запит: making computers understand human emotions from text
        ============================================================

        --- BM25: 'making computers understand human emotions from text' ---
        [17.0228] Modeling the Experience of Emotion
        [15.9082] Emotion capture based on body postures and movements
        [15.8738] Faith in the Algorithm, Part 1: Beyond the Turing Test
        [15.2162] Text as Statistical Mechanics Object
        [14.8250] Identification of parameters underlying emotions and a class

        --- Векторний: 'making computers understand human emotions from text' ---
        [0.6255] Modeling the Experience of Emotion
        [0.6157] Tagging multimedia stimuli with ontologies
        [0.5711] Syst\`emes interactifs sensibles aux \'emotions : architectu
        [0.5546] Emotion capture based on body postures and movements
        [0.5284] A computational model of affects

        --- Гібридний RRF (k=60): 'making computers understand human emotions from text' ---
        [RRF=0.01639] Modeling the Experience of Emotion
        [RRF=0.01639] Modeling the Experience of Emotion
        [RRF=0.01613] Emotion capture based on body postures and movements
        [RRF=0.01613] Tagging multimedia stimuli with ontologies
        [RRF=0.01587] Faith in the Algorithm, Part 1: Beyond the Turing Test

        --- Гібридний RRF (k=1): 'making computers understand human emotions from text' ---
        [RRF=0.50000] Modeling the Experience of Emotion
        [RRF=0.50000] Modeling the Experience of Emotion
        [RRF=0.33333] Emotion capture based on body postures and movements
        [RRF=0.33333] Tagging multimedia stimuli with ontologies
        [RRF=0.25000] Faith in the Algorithm, Part 1: Beyond the Turing Test