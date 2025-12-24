PHIẾU ĐĂNG KÝ DỰ THI
HỘI THI TÌM KIẾM TÀI NĂNG CNTT NĂM 2025 – LẦN THỨ 10

I. THÔNG TIN THÍ SINH
TT	MSSV	Họ và tên	Ngày sinh	Trường, Lớp	Email	Điện thoại
1						
2						
3						
4						


II. THÔNG TIN ĐỀ TÀI DỰ THI:
1. Bảng dự thi đăng ký (Chọn 1 trong các bảng thi sau): 
☐	Bảng A: An toàn thông tin / Information Security
☐	Bảng B: Ứng dụng trên thiết bị thông minh / Smart Device Applications
☐	Bảng C: Ứng dụng Website / Website Applications
☑	Bảng D: Trí tuệ nhân tạo & Công nghệ Chuỗi khối / AI & Blockchain
☐	Bảng E: Mạng máy tính / Computer Networks
☐	Bảng F: Khoa học Dữ liệu/ Data Science

2. Thông tin đề tài dự thi:

2.1. Tên đề tài dự thi: 
	Ứng dụng AI trong chơi cờ vây
	
2.2. Nội dung và ý tưởng: 
	Cờ Vây là một trong những game cờ phức tạp nhất với không gian trạng thái lên đến 10^170, 
	đòi hỏi AI phải có khả năng tư duy chiến lược sâu sắc. Dự án GoGame là một hệ thống học tập 
	Cờ Vây thông minh, không chỉ là game giải trí mà còn là platform giáo dục với AI coach 
	mạnh mẽ. Hệ thống sử dụng kết hợp các thuật toán AI cổ điển (Minimax, MCTS) với Deep Learning 
	(Policy Network, Value Network) để tạo ra một AI có khả năng chơi ở nhiều cấp độ, đồng thời 
	cung cấp các tính năng phân tích và gợi ý để giúp người học cải thiện kỹ năng. Ý tưởng xuất 
	phát từ nhu cầu thực tế: người học Cờ Vây cần công cụ để phân tích vị trí, học hỏi từ sai lầm, và có đối thủ AI phù hợp với trình độ để luyện tập.
	
2.3. Cơ sở lý thuyết và công nghệ sử dụng: 
	
	2.3.1. Thuật toán AI (C++ Engine):
	- Minimax với Alpha-Beta Pruning: Thuật toán tìm kiếm đối kháng cổ điển, sử dụng 
	  depth-limited search (1-4 ply tùy level). Alpha-Beta pruning giảm độ phức tạp từ O(b^d) 
	  xuống O(b^(d/2)) bằng cách cắt tỉa các nhánh không cần thiết. Move ordering tối ưu 
	  thứ tự xét nước đi để tăng hiệu quả pruning.
	- Monte Carlo Tree Search (MCTS): Thuật toán hiện đại dựa trên 4 pha: Selection (UCB1), 
	  Expansion, Simulation (random playout), Backpropagation. Không cần evaluation function 
	  hoàn hảo, tự học từ simulations. Có thể kết hợp với Policy Network để guide search.
	- Transposition Table: Cache các vị trí đã tính toán để tránh tính lại, sử dụng Zobrist 
	  hashing cho O(1) lookup.
	- Opening Book: Database các opening sequences từ professional games, giúp AI chơi khai 
	  cuộc chuyên nghiệp.
	- Evaluation Function: Đánh giá vị trí dựa trên territory estimation, group strength, 
	  pattern matching, và prisoners count.
	
	2.3.2. Machine Learning (PyTorch):
	- Policy Network: Convolutional Neural Network (CNN) nhận input 17 feature planes 
	  (stone colors, liberties, history), output probability distribution over 361 moves. 
	  Được train từ professional games với supervised learning, dùng cho hint system và 
	  MCTS guidance.
	- Value Network: CNN tương tự nhưng output win probability (0-1) thay vì move distribution. 
	  Đánh giá vị trí hiện tại, dùng cho position analysis và MCTS evaluation.
	- Multi-Task Learning Model: Shared backbone (ResNet-like với 4 residual blocks) + 3 task 
	  heads: Threat Head (heatmap phát hiện mối đe dọa), Attack Head (heatmap cơ hội tấn công), 
	  Intent Head (classification ý định đối thủ). Multi-task learning giúp models học tốt 
	  hơn vì share features, tăng generalization.
	- Training Pipeline: Parse SGF files từ professional games → extract positions và labels 
	  → supervised learning với multi-task loss → evaluation trên held-out games → deployment.
	
	2.3.3. Kiến trúc Hybrid C++/Python:
	- C++20: AI Engine core (board logic, minimax, MCTS) - performance-critical, xử lý hàng 
	  nghìn simulations mỗi nước. Sử dụng bitboard representation, Zobrist hashing, memory 
	  pool optimization.
	- Python 3.10+: ML models (PyTorch 2.0+), training pipeline, high-level orchestration. 
	  Ecosystem ML mạnh, dễ phát triển và experiment.
	- pybind11: Python-C++ binding, expose C++ classes/functions sang Python. Python gọi C++ 
	  engine cho performance, C++ callback Python cho neural network inference.
	- Lý do hybrid: Cờ Vây có branching factor ~250, cần xử lý nhanh. C++ cho hot path 
	  (50-100x faster), Python cho ML và flexibility.
	
	2.3.4. Backend (FastAPI):
	- Framework: FastAPI 0.111.0 - async REST API, tự động generate OpenAPI docs, type hints 
	  với Pydantic.
	- Database: PostgreSQL 14+ (SQLAlchemy 2.0) cho structured data (users, matches, coins), 
	  MongoDB 6+ (Motor async driver) cho game states, SGF records, ML analysis.
	- Authentication: JWT với Access + Refresh tokens, Argon2 password hashing (memory-hard 
	  để chống brute-force).
	- Real-time: WebSocket cho game synchronization, matchmaking queue updates.
	- Migration: Alembic cho database schema versioning.
	
	2.3.5. Frontend (React):
	- Framework: React 18.2.0 với hooks, context API cho state management.
	- Build Tool: Vite 7.2.4 - fast HMR, optimized production builds.
	- HTTP Client: Axios 1.6.0 cho API calls, interceptors cho auth.
	- Routing: React Router DOM 6.20.0 cho SPA navigation.
	- UI Libraries: Framer Motion cho animations, React Icons cho icons.
	- WebSocket: Native WebSocket API cho real-time game updates.
	
	2.3.6. Deployment & Infrastructure:
	- Containerization: Docker cho consistent environments, docker-compose cho local development.
	- Cloud Deployment: Fly.io (backend), có thể deploy lên AWS, Azure, GCP.
	- CI/CD: GitHub Actions cho automated testing và deployment.
	- Monitoring: Logging, error tracking, performance metrics.
	
2.4. Chức năng chính của sản phẩm: 
	
	2.4.1. Game Modes:
	- PvAI (Player vs AI): Chơi với AI có 4 cấp độ:
	  • Level 1 (Beginner): Minimax depth 1, 15% random moves, timeout 5s, phù hợp người mới
	  • Level 2 (Intermediate): Minimax depth 2, alpha-beta pruning, move ordering, opening 
	    book, timeout 8s
	  • Level 3 (Hard): Minimax depth 2-3, tất cả optimizations, transposition table, timeout 
	    6-15s
	  • Level 4 (Expert): Minimax depth 3-4, advanced move ordering, timeout 10-20s
	- PvP (Player vs Player): Chơi với người chơi khác qua internet, real-time synchronization
	- Room Code: Tạo phòng riêng với mã 6 ký tự, bạn bè có thể tham gia trực tiếp
	
	2.4.2. Matchmaking System:
	- Automatic Matching: Tự động ghép trận dựa trên ELO rating và board size preference
	- ELO-based Matching: Tìm đối thủ có ELO trong khoảng ±200 điểm để đảm bảo công bằng
	- Ready System: Cả 2 players phải confirm ready mới bắt đầu game
	- Queue Management: Real-time queue status, có thể cancel và rejoin
	
	2.4.3. Premium Features (AI-Powered):
	- Hint System (10 coins): Sử dụng Policy Network để gợi ý top 3 nước đi tốt nhất với:
	  • Xác suất cho mỗi nước đi
	  • Lý do tại sao nước đi tốt (capture, save group, expand territory, etc.)
	  • Visualization trên board
	- Position Analysis (20 coins): Phân tích chi tiết vị trí hiện tại với:
	  • Win Probability: Value Network đánh giá ai đang thắng (0-100%)
	  • Territory Estimation: Ước tính đất của mỗi bên
	  • Threat Heatmap: Multi-Task Model phát hiện các mối đe dọa (nhóm yếu, atari)
	  • Attack Heatmap: Cơ hội tấn công, invasion points
	  • Intent Prediction: Dự đoán ý định của đối thủ (defend, attack, expand)
	- Game Review (30 coins): Phân tích toàn bộ ván đấu sau khi kết thúc:
	  • Highlight mistakes: Các nước đi sai lầm với loss value
	  • Suggest improvements: Nước đi tốt hơn cho từng sai lầm
	  • Key moments: Các thời điểm quan trọng (capture, life/death fight, ko fight)
	  • Overall analysis: Win probability curve theo thời gian, accuracy score
	
	2.4.4. Ranking & Statistics:
	- ELO Rating System: Khởi đầu 1500, cập nhật sau mỗi game PvP dựa trên kết quả và ELO 
	  đối thủ. Công thức ELO chuẩn với K-factor = 32.
	- Leaderboard: Top 100 người chơi, có thể filter theo board size, time period
	- Personal Statistics: Win rate, total matches, win/loss/draw breakdown, average game 
	  length, favorite openings, mistake frequency
	- Match History: Lịch sử tất cả games với filters, có thể replay bất kỳ game nào
	
	2.4.5. Game Features:
	- Board Sizes: Hỗ trợ 9×9 và 19×19, có thể chọn khi tạo match
	- Time Control: Giới hạn thời gian cho PvP matches (ví dụ: 10 phút + 30s/move), tự động 
	  kết thúc khi hết thời gian
	- Undo Moves: Hoàn tác nước đi trong game (nếu đối thủ đồng ý)
	- Ko Detection: Tự động phát hiện và cảnh báo khi có Ko rule violation
	- Move History: Hiển thị tất cả nước đi với timeline, có thể navigate qua các nước
	- SGF Support: Lưu và load games theo format SGF chuẩn, có thể import/export
	
	2.4.6. Coin System:
	- Earning Coins: Daily login bonus (10 coins), win game (10 coins), complete game (5 coins), 
	  rank up (50 coins), achievements (20 coins)
	- Spending Coins: Premium features (hint 10, analysis 20, review 30 coins)
	- Transaction History: Lịch sử tất cả giao dịch với filters
	
	2.4.7. User Interface:
	- Interactive Game Board: Click để đánh cờ, hover để preview move, drag để navigate
	- ML Visualization: Overlay heatmaps từ Multi-Task Model (threat, attack), color-coded 
	  territory estimation
	- Responsive Design: Hoạt động tốt trên desktop và tablet
	- Real-time Updates: WebSocket cho instant game state synchronization
	- Dialogs: Login/Register, Matchmaking, Settings, Shop, Premium Features
	
	2.4.8. AI Engine với Heuristic Chuyên Nghiệp (Đang triển khai):
	- Fuseki (Khai Cuộc): Heuristic thông minh cho opening với corner-first principle, side 
	  expansion, board balance, influence spread. Đánh giá vị trí theo nguyên tắc chuyên nghiệp.
	- Joseki (Định Thức Góc): Database joseki patterns với context-aware evaluation. AI hiểu 
	  khi nào nên chơi joseki và khi nào không nên.
	- Tesuji (Nước Đi Tinh Tế): Phát hiện và đánh giá tactical moves (atari, hane, cut, 
	  connection, peep, attachment, shoulder hit) với giá trị khác nhau tùy context.
	- Life and Death Analysis: Phân tích sâu về life status (alive, dead, unsettled, in atari), 
	  đánh giá khả năng giết/cứu nhóm.
	- Ko Fights Evaluation: Đánh giá giá trị của ko, ko threats, sente/gote trong ko.
	- Sente/Gote Recognition: Phân biệt sente moves (đối thủ phải đáp) và gote moves, ưu tiên 
	  sente moves.
	- Aji (Tiềm Năng) Assessment: Đánh giá tiềm năng của quân cờ, aji được tạo ra và bị phá hủy.
	- Thickness/Thinness Evaluation: Đánh giá độ dày/mỏng của nhóm.
	- Endgame (Yose) Heuristic: Focus vào sente moves, territory gain, reduction, invasion.
	- Shape Judgment: Đánh giá chất lượng shape (good shape vs bad shape).
	
	2.4.9. Hệ Thống Xã Hội (Đang triển khai):
	- Chat System: Real-time chat trong và ngoài game với typing indicators, read receipts, 
	  emoji support. Tích hợp Firebase Realtime Database hoặc SendBird.
	- Friends System: Kết bạn, gửi lời mời, xem online status, challenge friends trực tiếp.
	- Forums & Learning Groups: Diễn đàn thảo luận, nhóm học tập, sharing game records, 
	  collaborative analysis.
	
	2.4.10. Tournament & Competitive (Đang triển khai):
	- Tournament System: Tổ chức giải đấu online với các format (single elimination, swiss, 
	  round robin). Entry fee bằng coins, prize pool distribution cho top players.
	- Betting System: Đặt cược coins vào các matches trong tournament, odds dựa trên ELO rating, 
	  payout khi thắng cược.
	- Advanced Rankings: Ngoài ELO, có dan rankings (kyu/dan), seasonal rankings, tournament 
	  rankings.
	
	2.4.11. Advanced Training Features (Đang triển khai):
	- Puzzle Mode: Tsumego puzzles với AI hints và solutions, difficulty levels từ beginner đến 
	  expert.
	- Training Mode: Focused training cho specific skills (opening, middle game, endgame, 
	  life/death) với AI coach feedback.
	- AI vs AI Mode: Xem AI chơi với nhau, học từ AI strategies, so sánh different AI levels.
	- Video Replay: Tạo video replay của games với narration, có thể share lên YouTube hoặc 
	  social media.
	
2.5. Tính sáng tạo và khả năng ứng dụng, thương mại hóa:
	Tính sáng tạo:
	- Kết hợp AI cổ điển (Minimax/MCTS) với Deep Learning trong một kiến trúc hybrid tối ưu
	- Multi-Task Learning Model phát hiện threat, attack, và intent - một cách tiếp cận mới 
	  trong phân tích Cờ Vây
	- Focus vào giáo dục thay vì chỉ gameplay, biến game thành công cụ học tập
	- Premium features sử dụng ML để cung cấp insights sâu cho người học
	
	Khả năng ứng dụng và thương mại hóa:
	- Ứng dụng trong giáo dục: Công cụ học Cờ Vây cho học sinh, sinh viên, người mới bắt đầu
	- EdTech market: Có thể tích hợp vào các nền tảng giáo dục, trung tâm dạy cờ
	- Monetization: Premium features (hint, analysis, review) với coin system hoặc subscription model
	- Scalable architecture: Có thể mở rộng để hỗ trợ nhiều người dùng đồng thời
	- Commercial viability: Sản phẩm đã deploy được, production-ready, có thể launch thương mại
	
2.6. Hướng phát triển trong tương lai: 
	
	2.6.1. AI Engine đạt cấp độ AlphaGo/AlphaZero:
	- Reinforcement Learning với Self-Play: Xây dựng hệ thống self-play training quy mô lớn, 
	  AI tự học từ hàng triệu games với chính nó. Sử dụng REINFORCE, TD-learning, hoặc 
	  Proximal Policy Optimization (PPO) để đạt level professional (9 dan+).
	- Transformer Architecture: Thay thế CNN bằng Transformer (như trong KataGo) để capture 
	  long-range dependencies tốt hơn, hiểu được global strategy.
	- Distributed Training: Training trên cluster với hàng trăm GPUs, có thể train models 
	  lớn hơn AlphaGo Lee (12 layers) lên 20-40 layers.
	- MCTS với Neural Network Guidance: Kết hợp MCTS với Policy/Value networks để đạt độ 
	  mạnh siêu việt, có thể đánh bại professional players.
	
	2.6.2. AI Research & Innovation:
	- Explainable AI (XAI): Phát triển models có thể giải thích reasoning một cách chi tiết, 
	  show attention maps, highlight các factors quan trọng trong decision-making. Giúp người 
	  học hiểu sâu hơn về AI thinking.
	- Adversarial Training: Train AI chống lại adversarial examples, tăng robustness và 
	  reliability.
	- Multi-Agent Learning: Nhiều AI agents học cùng nhau, share knowledge, compete và 
	  collaborate để cải thiện.
	- Meta-Learning: AI học cách học nhanh hơn, adapt nhanh với strategies mới.
	
	2.6.3. Platform Ecosystem & Integration:
	- GoGame API Platform: Mở rộng thành platform với public API, cho phép third-party 
	  developers xây dựng apps trên nền tảng GoGame. Ví dụ: custom training tools, analysis 
	  plugins, tournament organizers.
	- Integration với Major Platforms: Tích hợp với Steam, Epic Games, App Store, Google Play 
	  để tiếp cận hàng triệu users.
	- Cross-Platform Play: Người chơi trên mobile có thể chơi với người trên desktop, seamless 
	  experience.
	- Cloud Gaming: Stream game qua cloud, không cần download, chơi trên bất kỳ device nào.
	
	2.6.4. Educational Technology (EdTech) Expansion:
	- AI-Powered Curriculum: AI tự động generate curriculum dựa trên skill level và learning 
	  pace của từng học sinh. Adaptive learning path.
	- Virtual Go Teacher: AI teacher có thể giảng dạy như một giáo viên thật, giải thích 
	  concepts, answer questions, provide personalized feedback.
	- Integration với Schools: Partnership với các trường học, trung tâm giáo dục để đưa Cờ Vây 
	  vào chương trình học. Có thể dùng như môn học ngoại khóa hoặc môn chính thức.
	- Certification Program: Cấp chứng chỉ cho học sinh hoàn thành các level, có thể dùng 
	  trong portfolio hoặc CV.
	- Parent-Teacher Dashboard: Dashboard cho phụ huynh và giáo viên theo dõi progress của học 
	  sinh, xem reports chi tiết.
	
	2.6.5. Commercial & Business Model:
	- Subscription Tiers: Premium subscription với nhiều tiers (Basic, Pro, Enterprise) cho 
	  individuals và organizations.
	- B2B Solutions: Cung cấp white-label solution cho các trung tâm dạy cờ, clubs, organizations 
	  muốn có platform riêng.
	- Sponsorship & Advertising: Partnership với brands, tournaments, events để monetize.

	2.6.6. Advanced Analytics & Big Data:
	- Go Analytics Platform: Phân tích big data từ hàng triệu games để discover patterns, 
	  trends, optimal strategies. Publish research papers.
	- Predictive Analytics: Dự đoán kết quả games, player performance, market trends.
	- Player Behavior Analysis: Phân tích behavior của players để improve UX, retention, 
	  engagement.
	- Global Go Database: Xây dựng database lớn nhất thế giới về Go games, positions, strategies, 
	  accessible cho researchers và players.
	
	2.6.7. International Expansion & Localization:
	- Multi-language Full Support: Hỗ trợ đầy đủ 20+ ngôn ngữ (Vietnamese, English, Chinese, 
	  Japanese, Korean, Spanish, French, German, etc.) với professional translation.
	- Regional Tournaments: Tổ chức tournaments theo region (Asia, Europe, Americas) với prizes 
	  và sponsors.
	- Cultural Adaptation: Adapt UI/UX theo văn hóa từng region, respect local customs và 
	  preferences.
	- Partnership với Go Federations: Partnership với các Go federations quốc tế (IGF, EGF, etc.) 
	  để organize official tournaments.
	
	2.6.8. Research & Academic Contributions:
	- Open Source AI Models: Open source một số models để cộng đồng research có thể contribute 
	  và improve.
	- Academic Partnerships: Collaboration với universities để research AI, publish papers, 
	  contribute to Go AI community.
	- Go AI Benchmark: Tạo benchmark standard cho Go AI evaluation, giúp compare different 
	  approaches.
	- Contribution to Go Theory: Phát hiện strategies mới, contribute to Go theory thông qua 
	  AI analysis.
	
	2.6.9. Emerging Technologies Integration:
	- AR/VR Support: Virtual reality Go experience, play trên 3D board, immersive learning.
	- Voice AI Assistant: Voice assistant có thể guide players, explain moves, answer questions 
	  bằng giọng nói tự nhiên.
	- Blockchain Integration: Sử dụng blockchain cho tournament results verification, achievement 
	  certification, transparent prize distribution.
	- IoT Integration: Smart Go boards kết nối với platform, automatic move detection, real-time 
	  analysis.
	
	2.6.10. Social Impact & Community:
	- Go for Everyone Initiative: Program miễn phí cho học sinh, người khuyết tật, communities 
	  thiếu resources để học Cờ Vây.
	- Charity Tournaments: Tổ chức tournaments từ thiện, donate proceeds cho causes.
	- Go Education in Schools: Lobby để đưa Cờ Vây vào chương trình giáo dục chính thức ở 
	  nhiều quốc gia.
	- Community Building: Xây dựng community lớn mạnh với millions of active users, forums, 
	  events, meetups.
	
2.7. Màn hình, hình ảnh chính của ứng dụng (screenshots): 
	[Chèn screenshots: Game board, Hint system, Position analysis, Leaderboard, Statistics]


Tôi xin cam đoan đề tài dự thi này do tôi (chúng tôi) tự làm và lời khai trên là đúng sự thật. 
						TP. Hồ Chí Minh, ngày  tháng 12 năm 2025
Thí sinh đại diện đội


. . . . . . . . . . . . . . . . . . . . . . 
