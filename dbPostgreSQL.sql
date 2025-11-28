--
-- PostgreSQL database dump
--

\restrict 7NAGMsqxqNBfyLjcgX3ncodbhGcIazBeqRhqEPIRofAFxjvtuSYbTvrOezwJhUG

-- Dumped from database version 18.1
-- Dumped by pg_dump version 18.1

-- Started on 2025-11-25 20:42:28

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET transaction_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

--
-- TOC entry 2 (class 3079 OID 16389)
-- Name: uuid-ossp; Type: EXTENSION; Schema: -; Owner: -
--

CREATE EXTENSION IF NOT EXISTS "uuid-ossp" WITH SCHEMA public;


--
-- TOC entry 5099 (class 0 OID 0)
-- Dependencies: 2
-- Name: EXTENSION "uuid-ossp"; Type: COMMENT; Schema: -; Owner: 
--

COMMENT ON EXTENSION "uuid-ossp" IS 'generate universally unique identifiers (UUIDs)';


SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- TOC entry 226 (class 1259 OID 16520)
-- Name: alembic_version; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.alembic_version (
    version_num character varying(32) NOT NULL
);


ALTER TABLE public.alembic_version OWNER TO postgres;

--
-- TOC entry 223 (class 1259 OID 16462)
-- Name: coin_transactions; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.coin_transactions (
    id uuid DEFAULT public.uuid_generate_v4() NOT NULL,
    user_id uuid NOT NULL,
    amount integer NOT NULL,
    type character varying(32) NOT NULL,
    source character varying(64),
    created_at timestamp with time zone DEFAULT now() NOT NULL
);


ALTER TABLE public.coin_transactions OWNER TO postgres;

--
-- TOC entry 222 (class 1259 OID 16441)
-- Name: matches; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.matches (
    id uuid DEFAULT public.uuid_generate_v4() NOT NULL,
    black_player_id uuid,
    white_player_id uuid,
    ai_level integer,
    board_size integer DEFAULT 9 NOT NULL,
    result character varying(32),
    started_at timestamp with time zone DEFAULT now() NOT NULL,
    finished_at timestamp with time zone,
    sgf_id character varying(64),
    premium_analysis_id character varying(64),
    room_code character varying(6),
    time_control_minutes integer,
    black_time_remaining_seconds integer,
    white_time_remaining_seconds integer,
    last_move_at timestamp with time zone,
    black_elo_change integer,
    white_elo_change integer,
    black_ready boolean DEFAULT false NOT NULL,
    white_ready boolean DEFAULT false NOT NULL
);


ALTER TABLE public.matches OWNER TO postgres;

--
-- TOC entry 5100 (class 0 OID 0)
-- Dependencies: 222
-- Name: COLUMN matches.room_code; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON COLUMN public.matches.room_code IS 'Mã bàn 6 ký tự để tham gia PvP match';


--
-- TOC entry 5101 (class 0 OID 0)
-- Dependencies: 222
-- Name: COLUMN matches.time_control_minutes; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON COLUMN public.matches.time_control_minutes IS 'Thời gian tổng cho mỗi người chơi (phút)';


--
-- TOC entry 5102 (class 0 OID 0)
-- Dependencies: 222
-- Name: COLUMN matches.black_time_remaining_seconds; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON COLUMN public.matches.black_time_remaining_seconds IS 'Thời gian còn lại của Black (giây)';


--
-- TOC entry 5103 (class 0 OID 0)
-- Dependencies: 222
-- Name: COLUMN matches.white_time_remaining_seconds; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON COLUMN public.matches.white_time_remaining_seconds IS 'Thời gian còn lại của White (giây)';


--
-- TOC entry 5104 (class 0 OID 0)
-- Dependencies: 222
-- Name: COLUMN matches.last_move_at; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON COLUMN public.matches.last_move_at IS 'Thời điểm nước đi cuối cùng';


--
-- TOC entry 225 (class 1259 OID 16504)
-- Name: model_versions; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.model_versions (
    id uuid DEFAULT public.uuid_generate_v4() NOT NULL,
    policy_path character varying(255),
    value_path character varying(255),
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    description text
);


ALTER TABLE public.model_versions OWNER TO postgres;

--
-- TOC entry 224 (class 1259 OID 16479)
-- Name: premium_requests; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.premium_requests (
    id uuid DEFAULT public.uuid_generate_v4() NOT NULL,
    user_id uuid NOT NULL,
    match_id uuid NOT NULL,
    feature character varying(32) NOT NULL,
    cost integer NOT NULL,
    status character varying(32) DEFAULT 'pending'::character varying NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    completed_at timestamp with time zone
);


ALTER TABLE public.premium_requests OWNER TO postgres;

--
-- TOC entry 221 (class 1259 OID 16422)
-- Name: refresh_tokens; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.refresh_tokens (
    id uuid DEFAULT public.uuid_generate_v4() NOT NULL,
    user_id uuid NOT NULL,
    token text NOT NULL,
    expires_at timestamp with time zone NOT NULL,
    revoked boolean DEFAULT false NOT NULL
);


ALTER TABLE public.refresh_tokens OWNER TO postgres;

--
-- TOC entry 220 (class 1259 OID 16400)
-- Name: users; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.users (
    id uuid DEFAULT public.uuid_generate_v4() NOT NULL,
    username character varying(32) NOT NULL,
    email character varying(255) NOT NULL,
    password_hash character varying(255) NOT NULL,
    elo_rating integer DEFAULT 1500 NOT NULL,
    coins integer DEFAULT 0 NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    last_login timestamp with time zone,
    display_name character varying(64),
    avatar_url character varying(255),
    preferences jsonb
);


ALTER TABLE public.users OWNER TO postgres;

--
-- TOC entry 5093 (class 0 OID 16520)
-- Dependencies: 226
-- Data for Name: alembic_version; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.alembic_version (version_num) FROM stdin;
add_ready_status
\.


--
-- TOC entry 5090 (class 0 OID 16462)
-- Dependencies: 223
-- Data for Name: coin_transactions; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.coin_transactions (id, user_id, amount, type, source, created_at) FROM stdin;
2d3c65ee-00ca-4bd7-9a04-1d56093a4368	b6b6a32e-fcb0-4b2c-93f8-6c6264a75889	200	purchase	test_package	2025-11-19 23:29:33.699778+07
7bf3fc2f-bc2b-493a-8ea9-6d6d5cb347be	b6b6a32e-fcb0-4b2c-93f8-6c6264a75889	-10	spend	premium_hint	2025-11-19 23:29:34.150847+07
d8d0783c-20e8-4555-9590-78979d8084b4	b6b6a32e-fcb0-4b2c-93f8-6c6264a75889	-20	spend	premium_analysis	2025-11-19 23:29:34.308292+07
34d258c7-9486-4b67-9cd5-7adccde1706b	b6b6a32e-fcb0-4b2c-93f8-6c6264a75889	-30	spend	premium_review	2025-11-19 23:29:34.393676+07
dd066c1a-a912-41ce-8984-e35779b3d16b	b6b6a32e-fcb0-4b2c-93f8-6c6264a75889	-10	spend	premium_hint	2025-11-19 23:55:38.838675+07
74410a47-8086-4c8f-99e6-78c047a3962c	b6b6a32e-fcb0-4b2c-93f8-6c6264a75889	-20	spend	premium_analysis	2025-11-19 23:55:38.945094+07
e1c76b24-5c2f-40d4-93ac-330cacf2f9cd	b6b6a32e-fcb0-4b2c-93f8-6c6264a75889	-10	spend	premium_hint	2025-11-19 23:56:25.999935+07
b82ae0be-2687-4ddb-ba45-5f841df1b0f0	b6b6a32e-fcb0-4b2c-93f8-6c6264a75889	-20	spend	premium_analysis	2025-11-19 23:56:26.13881+07
90bbc563-bc72-43f8-9b41-47dcef454932	b6b6a32e-fcb0-4b2c-93f8-6c6264a75889	-10	spend	premium_hint	2025-11-19 23:56:40.481052+07
010970b8-12ab-4421-8346-ad384fc11836	b6b6a32e-fcb0-4b2c-93f8-6c6264a75889	-20	spend	premium_analysis	2025-11-19 23:56:40.495673+07
55eaa665-67cf-4848-b7bc-731d13d64ff5	b6b6a32e-fcb0-4b2c-93f8-6c6264a75889	-10	spend	premium_hint	2025-11-19 23:59:03.24884+07
123eccc5-6d0a-47ac-b624-e250d6768b59	b6b6a32e-fcb0-4b2c-93f8-6c6264a75889	-20	spend	premium_analysis	2025-11-19 23:59:03.362984+07
0e01171e-32c9-437d-8558-4d7f690647b1	b6b6a32e-fcb0-4b2c-93f8-6c6264a75889	100	purchase	test	2025-11-19 23:59:29.330401+07
bebef784-7eb0-4162-8460-b6be56a85115	b6b6a32e-fcb0-4b2c-93f8-6c6264a75889	-10	spend	premium_hint	2025-11-19 23:59:29.410346+07
b5a83878-9d47-44fa-a21d-0b50d7792375	b6b6a32e-fcb0-4b2c-93f8-6c6264a75889	-20	spend	premium_analysis	2025-11-19 23:59:29.507826+07
2b9ad20e-a593-4197-9ac3-a07f324a5e0b	b6b6a32e-fcb0-4b2c-93f8-6c6264a75889	200	purchase	test	2025-11-20 00:06:49.880479+07
a7b57c9b-21f5-4efc-82a3-fd52f3449c44	b6b6a32e-fcb0-4b2c-93f8-6c6264a75889	-10	spend	premium_hint	2025-11-20 00:06:49.946639+07
5091fffd-dcdb-42fc-95f9-67053b899911	b6b6a32e-fcb0-4b2c-93f8-6c6264a75889	-20	spend	premium_analysis	2025-11-20 00:06:50.033418+07
\.


--
-- TOC entry 5089 (class 0 OID 16441)
-- Dependencies: 222
-- Data for Name: matches; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.matches (id, black_player_id, white_player_id, ai_level, board_size, result, started_at, finished_at, sgf_id, premium_analysis_id, room_code, time_control_minutes, black_time_remaining_seconds, white_time_remaining_seconds, last_move_at, black_elo_change, white_elo_change, black_ready, white_ready) FROM stdin;
1d60439d-04a8-4d17-8459-0ea9c0fe9597	abfb01d6-b86b-44e8-85be-2ef2cca739b9	\N	1	9	\N	2025-11-19 22:10:23.196392+07	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	f	f
57727f87-3437-43c6-a38e-0e1c39b26202	abfb01d6-b86b-44e8-85be-2ef2cca739b9	\N	1	9	\N	2025-11-19 22:10:58.802392+07	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	f	f
dbc4564f-072a-42e0-8bc7-6bdb394284f8	b6b6a32e-fcb0-4b2c-93f8-6c6264a75889	\N	1	9	\N	2025-11-19 23:29:33.791916+07	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	f	f
2e74df82-99a9-410d-9bc9-0a021b167ef8	b6b6a32e-fcb0-4b2c-93f8-6c6264a75889	\N	1	9	\N	2025-11-19 23:41:41.265844+07	\N	2e74df82-99a9-410d-9bc9-0a021b167ef8	\N	\N	\N	\N	\N	\N	\N	\N	f	f
b8a9cd5f-1244-4992-bcc7-1ab1b401d5b1	b6b6a32e-fcb0-4b2c-93f8-6c6264a75889	\N	1	9	\N	2025-11-19 23:55:38.592735+07	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	f	f
6b135ed2-6d7b-40fa-9d54-81ed6aec457a	b6b6a32e-fcb0-4b2c-93f8-6c6264a75889	\N	1	9	\N	2025-11-19 23:56:25.692406+07	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	f	f
e8751b14-4165-42ff-b47c-c3cfb53c53e8	b6b6a32e-fcb0-4b2c-93f8-6c6264a75889	\N	1	9	\N	2025-11-19 23:56:40.44977+07	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	f	f
0febc7aa-4cab-41ee-87cf-fedfb10e774d	b6b6a32e-fcb0-4b2c-93f8-6c6264a75889	\N	1	9	\N	2025-11-19 23:59:02.470529+07	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	f	f
33dd9019-872d-4813-879e-e5fa94f21f20	b6b6a32e-fcb0-4b2c-93f8-6c6264a75889	\N	1	9	\N	2025-11-19 23:59:27.789994+07	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	f	f
7ff28137-c70c-4671-8c92-ac39bfcd1f61	b6b6a32e-fcb0-4b2c-93f8-6c6264a75889	\N	1	9	W+R	2025-11-20 00:01:49.58468+07	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	f	f
5f38069f-7c5c-4d23-a2b7-dc2136ec12a8	b6b6a32e-fcb0-4b2c-93f8-6c6264a75889	\N	1	9	\N	2025-11-20 00:01:51.947121+07	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	f	f
69dee122-3fad-4fbf-b9b1-772ae5df64b0	b6b6a32e-fcb0-4b2c-93f8-6c6264a75889	\N	1	9	\N	2025-11-20 00:01:48.744744+07	\N	69dee122-3fad-4fbf-b9b1-772ae5df64b0	\N	\N	\N	\N	\N	\N	\N	\N	f	f
02b72e9d-d27d-4f87-9529-69841bae0b8c	b6b6a32e-fcb0-4b2c-93f8-6c6264a75889	\N	1	9	\N	2025-11-20 00:01:52.495079+07	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	f	f
4582c655-f99b-4653-a00d-69c46216d01a	b6b6a32e-fcb0-4b2c-93f8-6c6264a75889	\N	2	9	\N	2025-11-20 00:01:53.27633+07	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	f	f
9a075770-86ac-4448-b577-0839f3ff4dec	b6b6a32e-fcb0-4b2c-93f8-6c6264a75889	\N	3	9	\N	2025-11-20 00:01:55.806088+07	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	f	f
ead02df3-76f9-4d6b-b5fa-162539ba4990	b6b6a32e-fcb0-4b2c-93f8-6c6264a75889	\N	4	9	\N	2025-11-20 00:02:26.103956+07	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	f	f
f6089860-5d69-463a-b994-2b2dc5f45997	b6b6a32e-fcb0-4b2c-93f8-6c6264a75889	\N	1	9	W+R	2025-11-20 00:03:59.530348+07	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	f	f
c3731eb7-7a0a-4899-b7ac-a7c87596e81f	b6b6a32e-fcb0-4b2c-93f8-6c6264a75889	\N	1	9	\N	2025-11-20 00:04:02.418194+07	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	f	f
089a55cd-07c7-43e5-9d51-5557a66dd467	b6b6a32e-fcb0-4b2c-93f8-6c6264a75889	\N	1	9	\N	2025-11-20 00:03:58.427847+07	\N	089a55cd-07c7-43e5-9d51-5557a66dd467	\N	\N	\N	\N	\N	\N	\N	\N	f	f
a284ad7b-30f1-4db3-b576-5af7d5ecee2f	b6b6a32e-fcb0-4b2c-93f8-6c6264a75889	\N	1	9	\N	2025-11-20 00:04:03.310164+07	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	f	f
fbdee37c-6f6a-4454-873a-392ff7d87d17	b6b6a32e-fcb0-4b2c-93f8-6c6264a75889	\N	2	9	\N	2025-11-20 00:04:04.155927+07	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	f	f
1ff31555-cd77-468a-ab95-54dbd2c41ed3	b6b6a32e-fcb0-4b2c-93f8-6c6264a75889	\N	3	9	\N	2025-11-20 00:04:07.158294+07	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	f	f
cc7465db-bffd-43b1-8b88-dcd9efc1796e	b6b6a32e-fcb0-4b2c-93f8-6c6264a75889	\N	4	9	\N	2025-11-20 00:04:37.505989+07	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	f	f
b7561df8-2c33-4a35-ab7e-13edef2ed73b	b6b6a32e-fcb0-4b2c-93f8-6c6264a75889	\N	2	9	\N	2025-11-20 00:06:15.548145+07	\N	b7561df8-2c33-4a35-ab7e-13edef2ed73b	\N	\N	\N	\N	\N	\N	\N	\N	f	f
25e34511-5728-47f8-977d-415ddb38a9b5	b6b6a32e-fcb0-4b2c-93f8-6c6264a75889	\N	\N	9	B+2.5	2025-11-20 00:26:26.576053+07	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	f	f
f61d760c-11c5-41bc-ba3a-9e098f8cd1c6	b6b6a32e-fcb0-4b2c-93f8-6c6264a75889	\N	1	9	\N	2025-11-20 00:26:26.70418+07	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	f	f
220a6a9a-6e9a-4e7c-93b2-b90bd998ae65	4ace5bbc-4f89-4a61-a936-36d24271af2b	\N	1	9	W+R	2025-11-20 04:36:35.803305+07	2025-11-20 04:46:59.795546+07	\N	\N	\N	\N	\N	\N	\N	\N	\N	f	f
df0075dd-6e56-4a12-9e9d-edc9632b0df6	66ccc111-bb61-484c-8485-1fe1b564019a	a368b186-fa80-4ad6-8b13-1b6d0bfa824d	\N	9	W+R	2025-11-20 00:30:22.528808+07	2025-11-20 00:30:24.025136+07	\N	\N	\N	\N	\N	\N	\N	\N	\N	f	f
18883a10-1ee4-4330-a5fb-8e0caeeeced8	b6b6a32e-fcb0-4b2c-93f8-6c6264a75889	\N	1	9	\N	2025-11-20 00:31:48.384626+07	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	f	f
8c60abaa-97bd-4ea0-a61b-9b537e600ca4	b6b6a32e-fcb0-4b2c-93f8-6c6264a75889	\N	2	9	\N	2025-11-20 00:31:49.321435+07	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	f	f
959b6ed9-e423-4e82-90f0-539f5a740e53	b6b6a32e-fcb0-4b2c-93f8-6c6264a75889	\N	3	9	\N	2025-11-20 00:31:51.755965+07	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	f	f
f503f182-1ca3-43e7-b55c-52a83af33b9a	b6b6a32e-fcb0-4b2c-93f8-6c6264a75889	\N	1	9	\N	2025-11-20 00:34:17.408777+07	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	f	f
ccf6fe9e-fd0f-463d-9fc7-ad7e7fdcbf5d	b6b6a32e-fcb0-4b2c-93f8-6c6264a75889	\N	2	9	\N	2025-11-20 00:34:18.067871+07	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	f	f
6faeed12-ccff-4748-9404-0399db70586f	b6b6a32e-fcb0-4b2c-93f8-6c6264a75889	\N	1	9	\N	2025-11-20 00:34:20.631411+07	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	f	f
6546d08a-9ba1-4a06-a405-537382ba8230	b6b6a32e-fcb0-4b2c-93f8-6c6264a75889	\N	1	9	\N	2025-11-20 00:34:21.672135+07	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	f	f
e35ca4c7-3eb1-44df-9dea-95b35cf61dc7	b6b6a32e-fcb0-4b2c-93f8-6c6264a75889	\N	\N	9	B+2.5	2025-11-20 00:34:22.01986+07	\N	e35ca4c7-3eb1-44df-9dea-95b35cf61dc7	\N	\N	\N	\N	\N	\N	\N	\N	f	f
febee185-3d25-4549-b3af-d3c978177002	b6b6a32e-fcb0-4b2c-93f8-6c6264a75889	\N	1	9	\N	2025-11-20 00:34:22.122863+07	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	f	f
6917dfe4-db56-480c-8deb-6944b2ba07bf	973324ab-d5b2-4e71-a94b-58124eb18aae	70def630-fdd8-476e-8103-d2c53b2990ca	\N	9	W+R	2025-11-20 00:34:25.186799+07	2025-11-20 00:34:25.782951+07	\N	\N	\N	\N	\N	\N	\N	\N	\N	f	f
a4ca9d75-d49b-4562-b41d-50b1e1c67f78	b6b6a32e-fcb0-4b2c-93f8-6c6264a75889	\N	1	9	\N	2025-11-20 00:34:57.88783+07	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	f	f
d22ea429-5d47-41b0-b00e-ba154759e3a6	b6b6a32e-fcb0-4b2c-93f8-6c6264a75889	\N	2	9	\N	2025-11-20 00:34:58.695975+07	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	f	f
6f390270-025d-43ea-9533-67cb0f6a205f	b6b6a32e-fcb0-4b2c-93f8-6c6264a75889	\N	1	9	\N	2025-11-20 00:35:01.502673+07	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	f	f
d7668ba2-4782-4407-bc5d-ea941de2e6ca	b6b6a32e-fcb0-4b2c-93f8-6c6264a75889	\N	1	9	\N	2025-11-20 00:35:02.653983+07	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	f	f
38885610-fe5a-42e0-a23b-641889ddcd75	b6b6a32e-fcb0-4b2c-93f8-6c6264a75889	\N	\N	9	B+2.5	2025-11-20 00:35:02.754181+07	\N	38885610-fe5a-42e0-a23b-641889ddcd75	\N	\N	\N	\N	\N	\N	\N	\N	f	f
36df3093-a905-4648-9c4c-691de454e1cd	b6b6a32e-fcb0-4b2c-93f8-6c6264a75889	\N	1	9	\N	2025-11-20 00:35:02.884388+07	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	f	f
978515bf-5d79-41ec-9516-1f7b5798455b	4ace5bbc-4f89-4a61-a936-36d24271af2b	\N	1	9	W+R	2025-11-20 04:46:59.804024+07	2025-11-20 04:49:40.150425+07	\N	\N	\N	\N	\N	\N	\N	\N	\N	f	f
92329ff8-a4c4-40e9-8413-c9aa681ca88b	973324ab-d5b2-4e71-a94b-58124eb18aae	70def630-fdd8-476e-8103-d2c53b2990ca	\N	9	W+R	2025-11-20 00:35:07.082351+07	2025-11-20 00:35:07.722103+07	\N	\N	\N	\N	\N	\N	\N	\N	\N	f	f
ae5eca92-5155-412a-8558-a92f930e5bf4	4ace5bbc-4f89-4a61-a936-36d24271af2b	\N	1	9	W+R	2025-11-20 03:33:25.903673+07	2025-11-20 03:34:39.245098+07	\N	\N	\N	\N	\N	\N	\N	\N	\N	f	f
0206b91b-7893-4961-90f6-b4efdffd4cb3	4ace5bbc-4f89-4a61-a936-36d24271af2b	\N	1	9	W+R	2025-11-20 03:17:04.350442+07	2025-11-20 04:13:15.783151+07	\N	\N	\N	\N	\N	\N	\N	\N	\N	f	f
8e16a13b-ec60-4eab-bca2-4f6014efc9df	4ace5bbc-4f89-4a61-a936-36d24271af2b	\N	1	9	W+R	2025-11-20 03:24:50.809447+07	2025-11-20 04:13:15.794317+07	\N	\N	\N	\N	\N	\N	\N	\N	\N	f	f
998e5cd7-ab96-46f6-a3de-41182019ddc0	4ace5bbc-4f89-4a61-a936-36d24271af2b	\N	1	9	W+R	2025-11-20 03:31:02.999539+07	2025-11-20 04:13:15.797819+07	\N	\N	\N	\N	\N	\N	\N	\N	\N	f	f
e8e6b949-90fd-42a7-95f7-61b05f1385c8	4ace5bbc-4f89-4a61-a936-36d24271af2b	\N	1	9	W+R	2025-11-20 03:38:18.345927+07	2025-11-20 04:13:15.801631+07	\N	\N	\N	\N	\N	\N	\N	\N	\N	f	f
a3645cdb-2fdf-4064-a3a8-55f5c9bfe33c	4ace5bbc-4f89-4a61-a936-36d24271af2b	\N	1	9	W+R	2025-11-20 03:42:42.765419+07	2025-11-20 04:13:15.804064+07	\N	\N	\N	\N	\N	\N	\N	\N	\N	f	f
325815b3-e0f4-47a7-ba40-12c8e41b7927	4ace5bbc-4f89-4a61-a936-36d24271af2b	\N	2	9	W+R	2025-11-20 03:53:00.44694+07	2025-11-20 04:13:15.806069+07	\N	\N	\N	\N	\N	\N	\N	\N	\N	f	f
3ba167bb-dbd4-4ffe-acd3-e7a26589e534	4ace5bbc-4f89-4a61-a936-36d24271af2b	\N	1	9	W+R	2025-11-20 03:57:54.088735+07	2025-11-20 04:13:15.808728+07	\N	\N	\N	\N	\N	\N	\N	\N	\N	f	f
0a318e97-77de-44e3-8a4e-b9fb3fb66b3f	4ace5bbc-4f89-4a61-a936-36d24271af2b	\N	3	9	W+R	2025-11-20 04:00:14.936815+07	2025-11-20 04:13:15.816155+07	\N	\N	\N	\N	\N	\N	\N	\N	\N	f	f
b39c1edc-1871-4811-be50-e69d8c673606	4ace5bbc-4f89-4a61-a936-36d24271af2b	\N	1	9	W+R	2025-11-20 04:13:15.819954+07	2025-11-20 04:17:09.101634+07	\N	\N	\N	\N	\N	\N	\N	\N	\N	f	f
5d099d27-d430-410b-9390-9ec8ee191ad1	4ace5bbc-4f89-4a61-a936-36d24271af2b	\N	1	9	W+R	2025-11-20 04:17:09.104452+07	2025-11-20 04:36:35.790386+07	\N	\N	\N	\N	\N	\N	\N	\N	\N	f	f
2de141b3-ee57-453e-bbf9-5d0128798a49	4ace5bbc-4f89-4a61-a936-36d24271af2b	\N	1	9	W+R	2025-11-20 04:49:40.153148+07	2025-11-20 04:55:36.260482+07	\N	\N	\N	\N	\N	\N	\N	\N	\N	f	f
0f45de5d-9d75-481e-8a2e-90344b4bb8ae	4ace5bbc-4f89-4a61-a936-36d24271af2b	\N	1	9	W+R	2025-11-20 04:55:36.263429+07	2025-11-20 05:15:53.194251+07	\N	\N	\N	\N	\N	\N	\N	\N	\N	f	f
ca1d440d-d54d-445b-b4af-3eaa557864ae	4ace5bbc-4f89-4a61-a936-36d24271af2b	\N	1	9	W+R	2025-11-20 05:15:53.198102+07	2025-11-20 05:18:12.591153+07	\N	\N	\N	\N	\N	\N	\N	\N	\N	f	f
d991c74f-9838-42ff-a047-d8b0b2285a70	4ace5bbc-4f89-4a61-a936-36d24271af2b	\N	1	9	W+R	2025-11-20 05:18:12.59717+07	2025-11-20 05:24:39.510697+07	\N	\N	\N	\N	\N	\N	\N	\N	\N	f	f
b83a9a84-42b5-494b-8586-3cf785f9b7b3	4ace5bbc-4f89-4a61-a936-36d24271af2b	\N	1	9	W+R	2025-11-20 05:24:39.518391+07	2025-11-20 05:33:36.749999+07	\N	\N	\N	\N	\N	\N	\N	\N	\N	f	f
7f263543-0f5b-4664-b368-92dee7179fc8	4ace5bbc-4f89-4a61-a936-36d24271af2b	\N	1	9	W+R	2025-11-20 05:33:36.763769+07	2025-11-20 05:37:31.078337+07	\N	\N	\N	\N	\N	\N	\N	\N	\N	f	f
7f3b05e1-758b-4263-ba8c-405178aad46e	4ace5bbc-4f89-4a61-a936-36d24271af2b	\N	1	9	W+R	2025-11-20 05:37:31.085185+07	2025-11-20 05:56:27.16412+07	\N	\N	\N	\N	\N	\N	\N	\N	\N	f	f
c073b50e-60e1-466d-bf32-519ec0323e6f	4ace5bbc-4f89-4a61-a936-36d24271af2b	\N	1	9	W+R	2025-11-20 05:59:06.739388+07	2025-11-20 06:01:43.993058+07	\N	\N	\N	\N	\N	\N	\N	\N	\N	f	f
13cf0436-fd57-4293-9b2b-332c8c63b5ca	1bebabf4-19b5-43d1-8677-c0dcd2bcc928	\N	\N	9	W+R	2025-11-20 06:16:10.220755+07	2025-11-20 06:22:06.237725+07	\N	\N	\N	\N	\N	\N	\N	\N	\N	f	f
502c54a8-a9bf-4867-8b35-c306bf8decf3	1bebabf4-19b5-43d1-8677-c0dcd2bcc928	\N	1	9	W+R	2025-11-20 06:22:06.243022+07	2025-11-20 06:28:46.375623+07	\N	\N	\N	\N	\N	\N	\N	\N	\N	f	f
dcc26dfe-228d-436d-8dba-011bbd34e4d3	1bebabf4-19b5-43d1-8677-c0dcd2bcc928	\N	1	9	W+R	2025-11-20 06:28:46.382676+07	2025-11-20 06:29:04.321948+07	\N	\N	\N	\N	\N	\N	\N	\N	\N	f	f
9e951427-ae36-48aa-a513-1ac7ae5334d8	1bebabf4-19b5-43d1-8677-c0dcd2bcc928	\N	1	9	W+R	2025-11-20 06:29:04.32626+07	2025-11-20 06:30:42.270013+07	\N	\N	\N	\N	\N	\N	\N	\N	\N	f	f
77adae8e-23dc-4413-91e9-2d4f506d9224	4ace5bbc-4f89-4a61-a936-36d24271af2b	\N	1	9	W+R	2025-11-20 06:01:44.001222+07	2025-11-20 18:42:05.905519+07	\N	\N	\N	\N	\N	\N	\N	\N	\N	f	f
d2690ddc-27c6-426b-9c5e-304149885f0c	4ace5bbc-4f89-4a61-a936-36d24271af2b	\N	1	9	W+R	2025-11-20 18:42:05.952745+07	2025-11-20 18:49:05.941396+07	\N	\N	\N	\N	\N	\N	\N	\N	\N	f	f
1a67e048-c5db-44e3-8f11-c71c6e57ed59	4ace5bbc-4f89-4a61-a936-36d24271af2b	\N	1	19	W+R	2025-11-20 18:49:12.15068+07	2025-11-20 18:51:23.368301+07	\N	\N	\N	\N	\N	\N	\N	\N	\N	f	f
08daa16e-8b9c-4fae-b3de-d9d99c01f2ca	4ace5bbc-4f89-4a61-a936-36d24271af2b	\N	1	9	W+R	2025-11-20 18:51:32.302818+07	2025-11-20 18:58:32.105281+07	\N	\N	\N	\N	\N	\N	\N	\N	\N	f	f
63fcc174-ed98-4258-ba3e-c34367a125e5	4ace5bbc-4f89-4a61-a936-36d24271af2b	\N	1	9	W+R	2025-11-20 18:58:32.111244+07	2025-11-20 18:58:36.564876+07	\N	\N	\N	\N	\N	\N	\N	\N	\N	f	f
8429286e-e3b8-4b8f-9944-7e1f136d7ffd	4ace5bbc-4f89-4a61-a936-36d24271af2b	\N	\N	9	W+R	2025-11-20 18:58:44.664507+07	2025-11-20 19:03:04.9428+07	\N	\N	\N	\N	\N	\N	\N	\N	\N	f	f
0def7030-59de-4d15-bfd1-61b0e21f6047	4ace5bbc-4f89-4a61-a936-36d24271af2b	\N	1	9	W+R	2025-11-20 19:03:04.956244+07	2025-11-20 19:03:44.927824+07	\N	\N	\N	\N	\N	\N	\N	\N	\N	f	f
d283dbba-8083-4600-accb-ee5be497927b	4ace5bbc-4f89-4a61-a936-36d24271af2b	\N	\N	9	W+R	2025-11-20 19:03:44.957972+07	2025-11-20 19:06:08.368987+07	\N	\N	\N	\N	\N	\N	\N	\N	\N	f	f
1863352e-9a7e-4a96-9452-2cc9446dd1d7	4ace5bbc-4f89-4a61-a936-36d24271af2b	\N	\N	9	W+R	2025-11-20 19:11:17.744616+07	2025-11-20 19:15:59.990729+07	\N	\N	\N	\N	\N	\N	\N	\N	\N	f	f
ba75d4cb-343e-490b-9dd6-8be19ad0705e	4ace5bbc-4f89-4a61-a936-36d24271af2b	\N	\N	9	W+R	2025-11-20 19:16:00.00741+07	2025-11-20 19:17:18.689322+07	\N	\N	\N	\N	\N	\N	\N	\N	\N	f	f
68651561-37ba-4676-966c-efe98ed55add	4ace5bbc-4f89-4a61-a936-36d24271af2b	\N	\N	9	W+R	2025-11-20 19:17:27.253444+07	2025-11-20 19:22:35.663513+07	\N	\N	\N	\N	\N	\N	\N	\N	\N	f	f
9bbc2851-10bf-4ac3-beca-ab29319430da	4ace5bbc-4f89-4a61-a936-36d24271af2b	\N	\N	9	W+R	2025-11-20 19:22:35.687491+07	2025-11-20 19:46:28.414726+07	\N	\N	\N	\N	\N	\N	\N	\N	\N	f	f
aaf4eb04-f1f8-4b70-962f-3e832e7c41c1	4ace5bbc-4f89-4a61-a936-36d24271af2b	\N	\N	9	W+R	2025-11-20 19:46:28.433698+07	2025-11-21 01:10:03.496851+07	\N	\N	\N	\N	\N	\N	\N	\N	\N	f	f
2b54aa24-c371-4b7e-8f1d-5ea227a4b98b	1bebabf4-19b5-43d1-8677-c0dcd2bcc928	\N	1	9	W+R	2025-11-20 06:30:42.27456+07	2025-11-21 06:33:51.269107+07	\N	\N	\N	\N	\N	\N	\N	\N	\N	f	f
f49b80ff-f787-4469-bddd-f0d23c157da7	4ace5bbc-4f89-4a61-a936-36d24271af2b	\N	\N	9	W+R	2025-11-21 01:10:03.518943+07	2025-11-21 01:24:04.113227+07	\N	\N	\N	\N	\N	\N	\N	\N	\N	f	f
4204bac5-b283-475f-b16d-66b4b8a8cc89	4ace5bbc-4f89-4a61-a936-36d24271af2b	\N	1	9	W+R	2025-11-21 01:24:04.133473+07	2025-11-21 01:32:52.496619+07	\N	\N	\N	\N	\N	\N	\N	\N	\N	f	f
7a93624c-4ade-45cc-bc9e-8f73f59b9518	4ace5bbc-4f89-4a61-a936-36d24271af2b	\N	1	9	W+R	2025-11-21 01:32:52.520962+07	2025-11-21 01:41:43.779315+07	\N	\N	\N	\N	\N	\N	\N	\N	\N	f	f
7620b1f4-b01d-46d2-9282-3eecac11b517	4ace5bbc-4f89-4a61-a936-36d24271af2b	\N	1	9	W+R	2025-11-21 01:41:43.795013+07	2025-11-21 02:00:47.694182+07	\N	\N	\N	\N	\N	\N	\N	\N	\N	f	f
1f2ab089-160e-44bb-a53a-de05ad8666e8	4ace5bbc-4f89-4a61-a936-36d24271af2b	\N	1	9	W+R	2025-11-21 02:00:47.711896+07	2025-11-21 03:49:57.178137+07	\N	\N	\N	\N	\N	\N	\N	\N	\N	f	f
69b72111-1779-4bef-a149-68fa314d1ff6	4ace5bbc-4f89-4a61-a936-36d24271af2b	\N	1	9	W+R	2025-11-21 03:50:13.483295+07	2025-11-21 04:13:46.443915+07	\N	\N	\N	\N	\N	\N	\N	\N	\N	f	f
017a0152-2c2e-44b0-941b-9f6010b8b5a2	1bebabf4-19b5-43d1-8677-c0dcd2bcc928	\N	1	9	W+R	2025-11-21 06:33:51.289626+07	2025-11-21 06:34:11.523479+07	\N	\N	\N	\N	\N	\N	\N	\N	\N	f	f
55047409-8c6c-4024-b0cd-48a35a37eda5	4ace5bbc-4f89-4a61-a936-36d24271af2b	\N	1	9	W+R	2025-11-21 04:13:46.455456+07	2025-11-21 06:52:11.431904+07	\N	\N	\N	\N	\N	\N	\N	\N	\N	f	f
303adb95-e80d-4669-97b1-eccfabda7d5a	4ace5bbc-4f89-4a61-a936-36d24271af2b	\N	1	9	W+R	2025-11-21 06:52:11.444136+07	2025-11-21 07:04:52.646398+07	\N	\N	\N	\N	\N	\N	\N	\N	\N	f	f
9e0decbb-1998-4b22-a997-d0f52bb20625	1bebabf4-19b5-43d1-8677-c0dcd2bcc928	\N	1	9	W+R	2025-11-21 06:34:11.536209+07	2025-11-21 07:56:28.56835+07	\N	\N	\N	\N	\N	\N	\N	\N	\N	f	f
959a3f12-cd90-4a8c-860f-0ce9a9396108	1bebabf4-19b5-43d1-8677-c0dcd2bcc928	\N	1	9	\N	2025-11-21 07:56:28.572936+07	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	f	f
748b6d14-481f-4c00-979d-bc985db647df	4ace5bbc-4f89-4a61-a936-36d24271af2b	\N	1	9	W+R	2025-11-21 07:04:52.65922+07	2025-11-21 16:29:38.745934+07	\N	\N	\N	\N	\N	\N	\N	\N	\N	f	f
ac1890da-52ad-4c2f-9c26-10d4e5b08aac	4ace5bbc-4f89-4a61-a936-36d24271af2b	\N	\N	9	W+R	2025-11-21 16:29:38.770905+07	2025-11-21 16:30:12.009268+07	\N	\N	\N	\N	\N	\N	\N	\N	\N	f	f
9edf4182-800e-4e9f-b333-a7e075ab93ce	4ace5bbc-4f89-4a61-a936-36d24271af2b	\N	\N	9	W+R	2025-11-21 16:30:12.01821+07	2025-11-21 16:34:22.923447+07	\N	\N	\N	\N	\N	\N	\N	\N	\N	f	f
e7a41eed-5534-4a6d-a6f0-071e7cf984ed	4ace5bbc-4f89-4a61-a936-36d24271af2b	\N	\N	9	W+R	2025-11-21 16:34:22.93862+07	2025-11-21 16:35:02.32653+07	\N	\N	\N	\N	\N	\N	\N	\N	\N	f	f
f5133aac-325e-4815-90a5-20cf16d4705f	4ace5bbc-4f89-4a61-a936-36d24271af2b	\N	\N	9	W+R	2025-11-21 16:35:02.335224+07	2025-11-21 16:36:54.759441+07	\N	\N	\N	\N	\N	\N	\N	\N	\N	f	f
c5f887ee-520c-4e66-b415-1af8db497dd6	4ace5bbc-4f89-4a61-a936-36d24271af2b	\N	\N	9	W+4	2025-11-21 16:36:54.770435+07	2025-11-21 16:41:44.099477+07	\N	\N	\N	\N	\N	\N	\N	\N	\N	f	f
be471430-cc63-46b4-9f2e-d9f5afaaaab0	4ace5bbc-4f89-4a61-a936-36d24271af2b	\N	1	9	W+R	2025-11-21 16:46:05.251517+07	2025-11-22 01:32:01.237124+07	\N	\N	\N	\N	\N	\N	\N	\N	\N	f	f
b3d20704-8b07-4430-afce-d70ed1506703	4ace5bbc-4f89-4a61-a936-36d24271af2b	\N	2	9	B+9	2025-11-22 01:32:01.253195+07	2025-11-22 02:25:23.424647+07	\N	\N	\N	\N	\N	\N	\N	\N	\N	f	f
d436746e-437e-4170-896e-5b52bd6e5f9e	4ace5bbc-4f89-4a61-a936-36d24271af2b	\N	1	9	W+1	2025-11-22 02:43:14.157494+07	2025-11-22 02:54:40.191194+07	\N	\N	\N	\N	\N	\N	\N	\N	\N	f	f
b654c180-17cd-48e4-b82a-62b74281cab9	4ace5bbc-4f89-4a61-a936-36d24271af2b	\N	1	9	W+R	2025-11-22 03:47:03.697459+07	2025-11-22 03:55:39.765169+07	\N	\N	\N	\N	\N	\N	\N	\N	\N	f	f
7a7a414a-7636-4387-bd5c-72965017d068	4ace5bbc-4f89-4a61-a936-36d24271af2b	\N	1	9	W+R	2025-11-22 03:55:39.795497+07	2025-11-22 04:02:06.433581+07	\N	\N	\N	\N	\N	\N	\N	\N	\N	f	f
f1cf7500-f8a6-4b43-9f8e-7188bdfae20c	4ace5bbc-4f89-4a61-a936-36d24271af2b	\N	1	9	W+3	2025-11-22 04:02:06.446202+07	2025-11-22 04:13:43.104425+07	\N	\N	\N	\N	\N	\N	\N	\N	\N	f	f
43e70fe5-3a31-4b68-89db-5c8c39b70f57	4ace5bbc-4f89-4a61-a936-36d24271af2b	\N	1	9	W+R	2025-11-22 04:17:47.762467+07	2025-11-22 05:02:16.955227+07	\N	\N	\N	\N	\N	\N	\N	\N	\N	f	f
6c13e661-3f42-44cc-af8d-6220db5330ed	4ace5bbc-4f89-4a61-a936-36d24271af2b	\N	1	19	W+R	2025-11-22 05:02:16.99642+07	2025-11-22 05:15:45.108768+07	\N	\N	\N	\N	\N	\N	\N	\N	\N	f	f
c37e96ae-e4b9-42da-a02f-4d2c9db83ea0	4ace5bbc-4f89-4a61-a936-36d24271af2b	\N	1	9	W+R	2025-11-22 05:15:45.150262+07	2025-11-22 05:31:02.09379+07	\N	\N	\N	\N	\N	\N	\N	\N	\N	f	f
ca0190f4-a4fd-416f-95ee-140fe15f62e4	4ace5bbc-4f89-4a61-a936-36d24271af2b	\N	1	9	B+19	2025-11-22 05:31:02.130379+07	2025-11-22 05:49:12.364644+07	\N	\N	\N	\N	\N	\N	\N	\N	\N	f	f
1946d46e-a716-49d8-8ff8-b930bec05a41	4ace5bbc-4f89-4a61-a936-36d24271af2b	\N	1	9	W+R	2025-11-22 05:49:23.695815+07	2025-11-23 19:20:00.139136+07	\N	\N	\N	\N	\N	\N	\N	\N	\N	f	f
c609e1b2-486f-42c8-b0b8-4dc3478f5b02	4ace5bbc-4f89-4a61-a936-36d24271af2b	\N	1	9	B+23	2025-11-23 19:20:00.161787+07	2025-11-23 19:34:24.397205+07	\N	\N	\N	\N	\N	\N	\N	\N	\N	f	f
574ad563-2f8b-4fd6-9fa0-79411d9427ce	4ace5bbc-4f89-4a61-a936-36d24271af2b	\N	1	9	W+R	2025-11-23 19:34:33.320335+07	2025-11-23 19:41:50.498073+07	\N	\N	\N	\N	\N	\N	\N	\N	\N	f	f
c11de74b-bca6-4f80-86d6-a19a3ec881d7	4ace5bbc-4f89-4a61-a936-36d24271af2b	\N	1	9	W+28	2025-11-23 19:41:50.519909+07	2025-11-23 19:51:24.861584+07	\N	\N	\N	\N	\N	\N	\N	\N	\N	f	f
002dc4d4-4771-4c2b-b397-2bd18dcc6703	4ace5bbc-4f89-4a61-a936-36d24271af2b	\N	1	9	W+R	2025-11-23 19:51:58.286663+07	2025-11-23 20:07:09.178728+07	\N	\N	\N	\N	\N	\N	\N	\N	\N	f	f
bbfd2514-cd14-453f-a519-471813b954b3	4ace5bbc-4f89-4a61-a936-36d24271af2b	\N	1	9	W+21	2025-11-23 20:07:09.192395+07	2025-11-23 20:09:41.89914+07	\N	\N	\N	\N	\N	\N	\N	\N	\N	f	f
5cd1a0a5-24ee-436e-849b-6aa588594c4c	4ace5bbc-4f89-4a61-a936-36d24271af2b	\N	1	9	W+R	2025-11-23 20:09:59.857313+07	2025-11-23 20:52:04.756726+07	\N	\N	\N	\N	\N	\N	\N	\N	\N	f	f
b85f276f-c1b5-4f87-9945-a7ac2d280e47	4ace5bbc-4f89-4a61-a936-36d24271af2b	\N	1	9	B+36	2025-11-23 20:52:04.776267+07	2025-11-23 21:11:58.993629+07	\N	\N	\N	\N	\N	\N	\N	\N	\N	f	f
da78d711-b8e4-4451-8c54-4090a4dd3efa	4ace5bbc-4f89-4a61-a936-36d24271af2b	\N	1	9	W+R	2025-11-23 21:12:51.488056+07	2025-11-23 21:34:09.504398+07	\N	\N	\N	\N	\N	\N	\N	\N	\N	f	f
c7468fa1-30c6-45e9-aea9-dfb6f102bc78	4ace5bbc-4f89-4a61-a936-36d24271af2b	\N	\N	9	W+R	2025-11-23 23:38:12.965312+07	2025-11-23 23:39:04.670144+07	\N	\N	\N	\N	\N	\N	\N	\N	\N	f	f
dc27b350-ef90-4c12-9312-dff8745e5fa3	4ace5bbc-4f89-4a61-a936-36d24271af2b	\N	\N	9	W+R	2025-11-24 00:06:32.656061+07	2025-11-24 00:08:43.966103+07	\N	\N	\N	\N	\N	\N	\N	\N	\N	f	f
3409acd6-fa6d-437b-b63b-62cabec008db	5dbe3908-51e7-4a85-91ed-1e5fd4c5a788	4ace5bbc-4f89-4a61-a936-36d24271af2b	\N	9	B+R	2025-11-24 02:27:25.413441+07	2025-11-24 02:38:44.20666+07	\N	\N	0SCR60	\N	\N	\N	\N	\N	\N	f	f
35218922-3e10-4627-88b1-4e5ea87cf440	4ace5bbc-4f89-4a61-a936-36d24271af2b	5dbe3908-51e7-4a85-91ed-1e5fd4c5a788	\N	9	B+R	2025-11-24 02:05:26.107817+07	2025-11-24 02:10:28.24089+07	\N	\N	B8G6JG	\N	\N	\N	\N	\N	\N	f	f
b8e2e0a9-d56f-4f34-b072-4853e93ff744	5dbe3908-51e7-4a85-91ed-1e5fd4c5a788	\N	\N	9	W+R	2025-11-24 02:18:25.476195+07	2025-11-24 02:27:25.400492+07	\N	\N	S6YF99	\N	\N	\N	\N	\N	\N	f	f
f7fee5f2-895b-4ebd-8e0e-3ac62e0693d5	4ace5bbc-4f89-4a61-a936-36d24271af2b	5dbe3908-51e7-4a85-91ed-1e5fd4c5a788	\N	9	B+R	2025-11-24 02:38:44.245088+07	2025-11-24 02:57:05.276564+07	\N	\N	FKTP8M	\N	\N	\N	\N	\N	\N	f	f
3e96665a-30a7-42b6-ac68-ce25b1f3330d	5dbe3908-51e7-4a85-91ed-1e5fd4c5a788	4ace5bbc-4f89-4a61-a936-36d24271af2b	\N	9	B+R	2025-11-25 03:55:53.21101+07	2025-11-25 10:05:51.967632+07	\N	\N	0YKSNX	10	600	600	2025-11-25 03:55:53.214647+07	12	-12	f	f
eb171fc1-a773-4b3b-98bb-b1c20660b389	5dbe3908-51e7-4a85-91ed-1e5fd4c5a788	\N	1	9	W+R	2025-11-25 02:23:46.356341+07	2025-11-25 02:29:05.492243+07	\N	\N	\N	\N	\N	\N	\N	\N	\N	f	f
0da22427-9879-434f-af8e-72dbebdd3ce8	4ace5bbc-4f89-4a61-a936-36d24271af2b	5dbe3908-51e7-4a85-91ed-1e5fd4c5a788	\N	9	B+3	2025-11-24 03:30:45.52625+07	2025-11-24 03:32:44.11557+07	\N	\N	7L5TDN	5	254	272	2025-11-24 03:32:05.976853+07	\N	\N	f	f
edf57557-07f6-47dc-ad46-d5e660c5e0ca	5dbe3908-51e7-4a85-91ed-1e5fd4c5a788	4ace5bbc-4f89-4a61-a936-36d24271af2b	\N	9	W+R	2025-11-25 02:24:00.99893+07	2025-11-25 02:29:05.516794+07	\N	\N	CTF3S6	10	600	600	2025-11-25 02:24:01.019061+07	-21	21	f	f
9559631c-c190-4656-9a1b-1fef7a1b610a	4ace5bbc-4f89-4a61-a936-36d24271af2b	5dbe3908-51e7-4a85-91ed-1e5fd4c5a788	\N	9	DRAW	2025-11-24 03:11:24.885217+07	2025-11-24 03:18:13.068146+07	\N	\N	HQKTPB	30	1687	1652	2025-11-24 03:15:51.735077+07	\N	\N	f	f
76061f54-0e5a-4841-8d41-65f4ccab0e55	4ace5bbc-4f89-4a61-a936-36d24271af2b	\N	\N	9	W+R	2025-11-24 03:25:41.447413+07	2025-11-24 03:30:45.511461+07	\N	\N	XH0TOG	30	1800	\N	2025-11-24 03:25:41.457575+07	\N	\N	f	f
3112a028-6808-44c2-880d-4a334061b050	5dbe3908-51e7-4a85-91ed-1e5fd4c5a788	4ace5bbc-4f89-4a61-a936-36d24271af2b	\N	9	W+R	2025-11-24 04:43:02.269916+07	2025-11-24 04:48:05.760076+07	\N	\N	SW6XOX	10	531	582	2025-11-24 04:44:30.672053+07	-25	25	f	f
1fb17a83-aec9-4c7c-82db-0aa6357c8574	4ace5bbc-4f89-4a61-a936-36d24271af2b	\N	1	9	W+R	2025-11-25 10:05:52.008515+07	2025-11-25 10:07:09.514242+07	\N	\N	\N	\N	\N	\N	\N	\N	\N	f	f
b991a884-1786-4619-bcab-16521247292c	5dbe3908-51e7-4a85-91ed-1e5fd4c5a788	4ace5bbc-4f89-4a61-a936-36d24271af2b	\N	9	W+R	2025-11-25 03:06:38.443516+07	2025-11-25 03:11:47.979558+07	\N	\N	OJEDIO	10	536	600	2025-11-25 03:07:43.303029+07	-21	21	t	t
a8d22d2b-928a-454c-877c-bf5c6bef978f	4ace5bbc-4f89-4a61-a936-36d24271af2b	\N	1	9	W+R	2025-11-25 02:29:25.78529+07	2025-11-25 10:05:51.950293+07	\N	\N	\N	\N	\N	\N	\N	\N	\N	f	f
\.


--
-- TOC entry 5092 (class 0 OID 16504)
-- Dependencies: 225
-- Data for Name: model_versions; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.model_versions (id, policy_path, value_path, created_at, description) FROM stdin;
\.


--
-- TOC entry 5091 (class 0 OID 16479)
-- Dependencies: 224
-- Data for Name: premium_requests; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.premium_requests (id, user_id, match_id, feature, cost, status, created_at, completed_at) FROM stdin;
457e9f57-5ff0-4d6c-a824-61ff4952e857	b6b6a32e-fcb0-4b2c-93f8-6c6264a75889	dbc4564f-072a-42e0-8bc7-6bdb394284f8	hint	10	completed	2025-11-19 23:29:34.17265+07	\N
d0d4226f-d5c6-4d59-86b3-de0dfd3ec9b7	b6b6a32e-fcb0-4b2c-93f8-6c6264a75889	dbc4564f-072a-42e0-8bc7-6bdb394284f8	analysis	20	completed	2025-11-19 23:29:34.331449+07	\N
88173996-2b6c-4860-b690-379b9de012bf	b6b6a32e-fcb0-4b2c-93f8-6c6264a75889	dbc4564f-072a-42e0-8bc7-6bdb394284f8	review	30	completed	2025-11-19 23:29:34.417182+07	\N
0e8552fd-698e-40ac-a4f5-62351d91743f	b6b6a32e-fcb0-4b2c-93f8-6c6264a75889	b8a9cd5f-1244-4992-bcc7-1ab1b401d5b1	hint	10	completed	2025-11-19 23:55:38.875791+07	\N
e6fe9d72-6264-4ab9-8b14-4e4ffad3bcbd	b6b6a32e-fcb0-4b2c-93f8-6c6264a75889	b8a9cd5f-1244-4992-bcc7-1ab1b401d5b1	analysis	20	completed	2025-11-19 23:55:38.96976+07	\N
4ddaf646-e51e-4f57-9691-3a244623402e	b6b6a32e-fcb0-4b2c-93f8-6c6264a75889	6b135ed2-6d7b-40fa-9d54-81ed6aec457a	hint	10	completed	2025-11-19 23:56:26.062805+07	\N
9f0e77c9-e5b2-40e2-8937-25e49102c128	b6b6a32e-fcb0-4b2c-93f8-6c6264a75889	6b135ed2-6d7b-40fa-9d54-81ed6aec457a	analysis	20	completed	2025-11-19 23:56:26.170232+07	\N
45a10dfe-e7d6-4e39-ad2f-6dfa36892800	b6b6a32e-fcb0-4b2c-93f8-6c6264a75889	e8751b14-4165-42ff-b47c-c3cfb53c53e8	hint	10	completed	2025-11-19 23:56:40.486557+07	\N
5bf80180-0b31-455b-893c-7b983af13027	b6b6a32e-fcb0-4b2c-93f8-6c6264a75889	e8751b14-4165-42ff-b47c-c3cfb53c53e8	analysis	20	completed	2025-11-19 23:56:40.500911+07	\N
86ae9736-853b-4672-96ca-f1f48975f5ae	b6b6a32e-fcb0-4b2c-93f8-6c6264a75889	0febc7aa-4cab-41ee-87cf-fedfb10e774d	hint	10	completed	2025-11-19 23:59:03.302519+07	\N
4de63b2e-1568-4a92-9942-16fe124ed872	b6b6a32e-fcb0-4b2c-93f8-6c6264a75889	0febc7aa-4cab-41ee-87cf-fedfb10e774d	analysis	20	completed	2025-11-19 23:59:03.393096+07	\N
adbb9768-dc4e-488f-920c-ade66a8f0122	b6b6a32e-fcb0-4b2c-93f8-6c6264a75889	33dd9019-872d-4813-879e-e5fa94f21f20	hint	10	completed	2025-11-19 23:59:29.449037+07	\N
fd7c83e0-2807-4141-bbe4-b21d0557893b	b6b6a32e-fcb0-4b2c-93f8-6c6264a75889	33dd9019-872d-4813-879e-e5fa94f21f20	analysis	20	completed	2025-11-19 23:59:29.529905+07	\N
46c0eddb-c165-4c51-9365-a6e45348cf8b	b6b6a32e-fcb0-4b2c-93f8-6c6264a75889	b7561df8-2c33-4a35-ab7e-13edef2ed73b	hint	10	completed	2025-11-20 00:06:49.978572+07	\N
349a1edf-82ef-4c47-8388-1e63695a5281	b6b6a32e-fcb0-4b2c-93f8-6c6264a75889	b7561df8-2c33-4a35-ab7e-13edef2ed73b	analysis	20	completed	2025-11-20 00:06:50.053365+07	\N
\.


--
-- TOC entry 5088 (class 0 OID 16422)
-- Dependencies: 221
-- Data for Name: refresh_tokens; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.refresh_tokens (id, user_id, token, expires_at, revoked) FROM stdin;
c5dce4d8-c673-4022-a280-eb5d9d4dfda6	abfb01d6-b86b-44e8-85be-2ef2cca739b9	eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJhYmZiMDFkNi1iODZiLTQ0ZTgtODViZS0yZWYyY2NhNzM5YjkiLCJqdGkiOiJjNWRjZTRkOC1jNjczLTQwMjItYTI4MC1lYjVkOWQ0ZGZkYTYiLCJ0eXBlIjoicmVmcmVzaCIsImlhdCI6MTc2MzU2NTAyMywiZXhwIjoxNzY0MTY5ODIzfQ.VVBETjdGN92MqeLgUfHW7M3uJVUaLYm-vQ1jAjwXCMI	2025-11-26 22:10:23.140908+07	f
4885922b-828f-46fc-8620-8909c2246746	abfb01d6-b86b-44e8-85be-2ef2cca739b9	eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJhYmZiMDFkNi1iODZiLTQ0ZTgtODViZS0yZWYyY2NhNzM5YjkiLCJqdGkiOiI0ODg1OTIyYi04MjhmLTQ2ZmMtODYyMC04OTA5YzIyNDY3NDYiLCJ0eXBlIjoicmVmcmVzaCIsImlhdCI6MTc2MzU2NTA1OCwiZXhwIjoxNzY0MTY5ODU4fQ.Nnr1bD7oVblim5w5CC19MIuxAHI8fgf8_CNXcLoEhvc	2025-11-26 22:10:58.786119+07	f
7c333c04-bffb-4be8-aeb0-98c0a79127f8	b6b6a32e-fcb0-4b2c-93f8-6c6264a75889	eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJiNmI2YTMyZS1mY2IwLTRiMmMtOTNmOC02YzYyNjRhNzU4ODkiLCJqdGkiOiI3YzMzM2MwNC1iZmZiLTRiZTgtYWViMC05OGMwYTc5MTI3ZjgiLCJ0eXBlIjoicmVmcmVzaCIsImlhdCI6MTc2MzU2OTcwNCwiZXhwIjoxNzY0MTc0NTA0fQ.w3YAeOe0mRQPhmJQJ_3OU28CMTzwoe22CuML5db8xWo	2025-11-26 23:28:24.21508+07	f
099352f5-4aa9-40d1-b2a7-a0657cccb7cf	b6b6a32e-fcb0-4b2c-93f8-6c6264a75889	eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJiNmI2YTMyZS1mY2IwLTRiMmMtOTNmOC02YzYyNjRhNzU4ODkiLCJqdGkiOiIwOTkzNTJmNS00YWE5LTQwZDEtYjJhNy1hMDY1N2NjY2I3Y2YiLCJ0eXBlIjoicmVmcmVzaCIsImlhdCI6MTc2MzU2OTc3MywiZXhwIjoxNzY0MTc0NTczfQ.PaSaY6aY3kiw_3GGZpRdwWiYWDdCIkgZrGz0d01i7BU	2025-11-26 23:29:33.619187+07	f
765c5394-5758-4d95-9f54-db090fd3a2a5	b6b6a32e-fcb0-4b2c-93f8-6c6264a75889	eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJiNmI2YTMyZS1mY2IwLTRiMmMtOTNmOC02YzYyNjRhNzU4ODkiLCJqdGkiOiI3NjVjNTM5NC01NzU4LTRkOTUtOWY1NC1kYjA5MGZkM2EyYTUiLCJ0eXBlIjoicmVmcmVzaCIsImlhdCI6MTc2MzU3MDQ5OSwiZXhwIjoxNzY0MTc1Mjk5fQ.0JFZCPeGLT6Cpdy65kDABQWvqdbIEqYdz1lIGrhDpgw	2025-11-26 23:41:39.92139+07	f
8eb23ca9-c9f9-436e-93a8-52b6318626bc	b6b6a32e-fcb0-4b2c-93f8-6c6264a75889	eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJiNmI2YTMyZS1mY2IwLTRiMmMtOTNmOC02YzYyNjRhNzU4ODkiLCJqdGkiOiI4ZWIyM2NhOS1jOWY5LTQzNmUtOTNhOC01MmI2MzE4NjI2YmMiLCJ0eXBlIjoicmVmcmVzaCIsImlhdCI6MTc2MzU3MTMzOCwiZXhwIjoxNzY0MTc2MTM4fQ.LQzLtDcXfN65bHaUCy31RrditYRFD1dTWZkAzHeI2LA	2025-11-26 23:55:38.545945+07	f
4aecfdc8-8c9d-4677-a112-52e73d2f99c4	b6b6a32e-fcb0-4b2c-93f8-6c6264a75889	eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJiNmI2YTMyZS1mY2IwLTRiMmMtOTNmOC02YzYyNjRhNzU4ODkiLCJqdGkiOiI0YWVjZmRjOC04YzlkLTQ2NzctYTExMi01MmU3M2QyZjk5YzQiLCJ0eXBlIjoicmVmcmVzaCIsImlhdCI6MTc2MzU3MTM4MCwiZXhwIjoxNzY0MTc2MTgwfQ.2731ItqM1O118OcHnCUHKCoAyj9gznq9vlELYyuaGlw	2025-11-26 23:56:20.278688+07	f
1cd01ab5-113d-4926-83a9-3458221cbd02	b6b6a32e-fcb0-4b2c-93f8-6c6264a75889	eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJiNmI2YTMyZS1mY2IwLTRiMmMtOTNmOC02YzYyNjRhNzU4ODkiLCJqdGkiOiIxY2QwMWFiNS0xMTNkLTQ5MjYtODNhOS0zNDU4MjIxY2JkMDIiLCJ0eXBlIjoicmVmcmVzaCIsImlhdCI6MTc2MzU3MTQwMCwiZXhwIjoxNzY0MTc2MjAwfQ.zHCBTPjlaTOUb6V2mZiiOud94yUZpp_S2E2EXGLLr6c	2025-11-26 23:56:40.442689+07	f
8e3cc339-8f0b-4f89-82b1-f12ecac90c61	b6b6a32e-fcb0-4b2c-93f8-6c6264a75889	eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJiNmI2YTMyZS1mY2IwLTRiMmMtOTNmOC02YzYyNjRhNzU4ODkiLCJqdGkiOiI4ZTNjYzMzOS04ZjBiLTRmODktODJiMS1mMTJlY2FjOTBjNjEiLCJ0eXBlIjoicmVmcmVzaCIsImlhdCI6MTc2MzU3MTUzNiwiZXhwIjoxNzY0MTc2MzM2fQ.2aTTUALjCqKhwNwXJftcf1HP_OTfe5pHH8nT9Mjni-c	2025-11-26 23:58:56.512802+07	f
f337354c-dfb6-40aa-9257-37c08895cb68	b6b6a32e-fcb0-4b2c-93f8-6c6264a75889	eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJiNmI2YTMyZS1mY2IwLTRiMmMtOTNmOC02YzYyNjRhNzU4ODkiLCJqdGkiOiJmMzM3MzU0Yy1kZmI2LTQwYWEtOTI1Ny0zN2MwODg5NWNiNjgiLCJ0eXBlIjoicmVmcmVzaCIsImlhdCI6MTc2MzU3MTU2MywiZXhwIjoxNzY0MTc2MzYzfQ.lVLA6kvMjYm_kI9w1gkRbDU-tAzfbhz3aZFQzL3IjD4	2025-11-26 23:59:23.630702+07	f
9069e143-9a90-4903-9c52-5957f9a10776	b6b6a32e-fcb0-4b2c-93f8-6c6264a75889	eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJiNmI2YTMyZS1mY2IwLTRiMmMtOTNmOC02YzYyNjRhNzU4ODkiLCJqdGkiOiI5MDY5ZTE0My05YTkwLTQ5MDMtOWM1Mi01OTU3ZjlhMTA3NzYiLCJ0eXBlIjoicmVmcmVzaCIsImlhdCI6MTc2MzU3MTcwMiwiZXhwIjoxNzY0MTc2NTAyfQ.FXL4kCqWvKpvnn9wJd5no4huHTI7Qdna--zcjNcjmxY	2025-11-27 00:01:42.919084+07	f
0de3821c-36c6-4e74-be1f-c053095af46f	b6b6a32e-fcb0-4b2c-93f8-6c6264a75889	eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJiNmI2YTMyZS1mY2IwLTRiMmMtOTNmOC02YzYyNjRhNzU4ODkiLCJqdGkiOiIwZGUzODIxYy0zNmM2LTRlNzQtYmUxZi1jMDUzMDk1YWY0NmYiLCJ0eXBlIjoicmVmcmVzaCIsImlhdCI6MTc2MzU3MTgyOSwiZXhwIjoxNzY0MTc2NjI5fQ.42PJd_Ka75B8Llhn-SoXFMhJsrn8wNWyTsn26tS1wu4	2025-11-27 00:03:49.947397+07	f
33ad0626-3b5d-4cc8-a610-101ecbb03d50	b6b6a32e-fcb0-4b2c-93f8-6c6264a75889	eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJiNmI2YTMyZS1mY2IwLTRiMmMtOTNmOC02YzYyNjRhNzU4ODkiLCJqdGkiOiIzM2FkMDYyNi0zYjVkLTRjYzgtYTYxMC0xMDFlY2JiMDNkNTAiLCJ0eXBlIjoicmVmcmVzaCIsImlhdCI6MTc2MzU3MTk2OSwiZXhwIjoxNzY0MTc2NzY5fQ.3vKqzPATz0j8VptGJ5hM8aXE83j2BCcTB0lYnnh8KJw	2025-11-27 00:06:09.761143+07	f
b5979a4d-ad22-4eb7-a97d-03c34f139557	b6b6a32e-fcb0-4b2c-93f8-6c6264a75889	eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJiNmI2YTMyZS1mY2IwLTRiMmMtOTNmOC02YzYyNjRhNzU4ODkiLCJqdGkiOiJiNTk3OWE0ZC1hZDIyLTRlYjctYTk3ZC0wM2MzNGYxMzk1NTciLCJ0eXBlIjoicmVmcmVzaCIsImlhdCI6MTc2MzU3MzE0NCwiZXhwIjoxNzY0MTc3OTQ0fQ.511ASu4AWMzXMBSSxWJfQZR3mBvpVXZ-5lPjgsV-kAQ	2025-11-27 00:25:44.299819+07	f
bacdbaa4-08f5-485c-9a36-3647accc97b3	b6b6a32e-fcb0-4b2c-93f8-6c6264a75889	eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJiNmI2YTMyZS1mY2IwLTRiMmMtOTNmOC02YzYyNjRhNzU4ODkiLCJqdGkiOiJiYWNkYmFhNC0wOGY1LTQ4NWMtOWEzNi0zNjQ3YWNjYzk3YjMiLCJ0eXBlIjoicmVmcmVzaCIsImlhdCI6MTc2MzU3MzE0NCwiZXhwIjoxNzY0MTc3OTQ0fQ.gjRivVktnGeH3RyrUbhT049nSioUWpfBWp8J0R34_V0	2025-11-27 00:25:44.314995+07	f
ed58b488-852c-471d-baf4-6a50502ba6a3	b6b6a32e-fcb0-4b2c-93f8-6c6264a75889	eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJiNmI2YTMyZS1mY2IwLTRiMmMtOTNmOC02YzYyNjRhNzU4ODkiLCJqdGkiOiJlZDU4YjQ4OC04NTJjLTQ3MWQtYmFmNC02YTUwNTAyYmE2YTMiLCJ0eXBlIjoicmVmcmVzaCIsImlhdCI6MTc2MzU3MzE0NCwiZXhwIjoxNzY0MTc3OTQ0fQ.989-lrIgbBtCKDQwrQFMpdD52s33ziOo0MYf94v3eXg	2025-11-27 00:25:44.323673+07	f
20adfba4-19a6-4f36-a8cd-8d8154309494	b6b6a32e-fcb0-4b2c-93f8-6c6264a75889	eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJiNmI2YTMyZS1mY2IwLTRiMmMtOTNmOC02YzYyNjRhNzU4ODkiLCJqdGkiOiIyMGFkZmJhNC0xOWE2LTRmMzYtYThjZC04ZDgxNTQzMDk0OTQiLCJ0eXBlIjoicmVmcmVzaCIsImlhdCI6MTc2MzU3MzE4NiwiZXhwIjoxNzY0MTc3OTg2fQ.9CuXBZSd__ZKOFYvwWMCZg3W-bQYG78aSyGXZIXHnZs	2025-11-27 00:26:26.354122+07	f
d7177cc0-1b28-4a76-913d-29fe48bf46ed	66ccc111-bb61-484c-8485-1fe1b564019a	eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI2NmNjYzExMS1iYjYxLTQ4NGMtODQ4NS0xZmUxYjU2NDAxOWEiLCJqdGkiOiJkNzE3N2NjMC0xYjI4LTRhNzYtOTEzZC0yOWZlNDhiZjQ2ZWQiLCJ0eXBlIjoicmVmcmVzaCIsImlhdCI6MTc2MzU3MzQyMiwiZXhwIjoxNzY0MTc4MjIyfQ.TUix0ywcY5i2nGvVWBaujgAoOfVFPXw3Tf1yByGpZ6Y	2025-11-27 00:30:22.16201+07	f
d824b93c-4ee3-42fd-92c9-6a69799b029a	a368b186-fa80-4ad6-8b13-1b6d0bfa824d	eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJhMzY4YjE4Ni1mYTgwLTRhZDYtOGIxMy0xYjZkMGJmYTgyNGQiLCJqdGkiOiJkODI0YjkzYy00ZWUzLTQyZmQtOTJjOS02YTY5Nzk5YjAyOWEiLCJ0eXBlIjoicmVmcmVzaCIsImlhdCI6MTc2MzU3MzQyMiwiZXhwIjoxNzY0MTc4MjIyfQ.cHsCg7NKZS5NcW03WbwtmYwcnlo_LdoWZIKIKFRCkDk	2025-11-27 00:30:22.381093+07	f
83cb5ff0-74c6-4ae5-8b20-0f69e420d8de	b6b6a32e-fcb0-4b2c-93f8-6c6264a75889	eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJiNmI2YTMyZS1mY2IwLTRiMmMtOTNmOC02YzYyNjRhNzU4ODkiLCJqdGkiOiI4M2NiNWZmMC03NGM2LTRhZTUtOGIyMC0wZjY5ZTQyMGQ4ZGUiLCJ0eXBlIjoicmVmcmVzaCIsImlhdCI6MTc2MzU3MzUwOCwiZXhwIjoxNzY0MTc4MzA4fQ.LZhvbWAZ0QIkG1AwvlTgsNAGn_aiO9a7E1ipJiwLoPY	2025-11-27 00:31:48.352457+07	f
6996c53a-e04c-4c34-a61e-b82adea41475	b6b6a32e-fcb0-4b2c-93f8-6c6264a75889	eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJiNmI2YTMyZS1mY2IwLTRiMmMtOTNmOC02YzYyNjRhNzU4ODkiLCJqdGkiOiI2OTk2YzUzYS1lMDRjLTRjMzQtYTYxZS1iODJhZGVhNDE0NzUiLCJ0eXBlIjoicmVmcmVzaCIsImlhdCI6MTc2MzU3MzY1NywiZXhwIjoxNzY0MTc4NDU3fQ.FEeW3ffK2furG3EGN5DHjW26oq0mF5jnW3JcYepM9vE	2025-11-27 00:34:17.367979+07	f
81d6c177-c5a9-403f-926a-0a5209f5a9d2	973324ab-d5b2-4e71-a94b-58124eb18aae	eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI5NzMzMjRhYi1kNWIyLTRlNzEtYTk0Yi01ODEyNGViMThhYWUiLCJqdGkiOiI4MWQ2YzE3Ny1jNWE5LTQwM2YtOTI2YS0wYTUyMDlmNWE5ZDIiLCJ0eXBlIjoicmVmcmVzaCIsImlhdCI6MTc2MzU3MzY2NCwiZXhwIjoxNzY0MTc4NDY0fQ.3jMa_nboymVXkipeWh8U97s5cPtFB38CjPaAEfnek6o	2025-11-27 00:34:24.900954+07	f
2394e4c0-e17f-435c-bbdd-70836cc1308a	70def630-fdd8-476e-8103-d2c53b2990ca	eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI3MGRlZjYzMC1mZGQ4LTQ3NmUtODEwMy1kMmM1M2IyOTkwY2EiLCJqdGkiOiIyMzk0ZTRjMC1lMTdmLTQzNWMtYmJkZC03MDgzNmNjMTMwOGEiLCJ0eXBlIjoicmVmcmVzaCIsImlhdCI6MTc2MzU3MzY2NSwiZXhwIjoxNzY0MTc4NDY1fQ.QWc5iOMHIBGilZVcgYuVzG5V9OnMt0-6kZZA14mpl-k	2025-11-27 00:34:25.089225+07	f
f5a1963d-a970-473a-88cc-30763c033b5a	b6b6a32e-fcb0-4b2c-93f8-6c6264a75889	eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJiNmI2YTMyZS1mY2IwLTRiMmMtOTNmOC02YzYyNjRhNzU4ODkiLCJqdGkiOiJmNWExOTYzZC1hOTcwLTQ3M2EtODhjYy0zMDc2M2MwMzNiNWEiLCJ0eXBlIjoicmVmcmVzaCIsImlhdCI6MTc2MzU3MzY5MCwiZXhwIjoxNzY0MTc4NDkwfQ.SY-zhiq4-HMCcqWj6-1qa22Bw9L83KPP6QJ_DVUORSI	2025-11-27 00:34:50.260495+07	f
f64e29a7-61ca-4b03-921d-47656bf7041c	973324ab-d5b2-4e71-a94b-58124eb18aae	eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI5NzMzMjRhYi1kNWIyLTRlNzEtYTk0Yi01ODEyNGViMThhYWUiLCJqdGkiOiJmNjRlMjlhNy02MWNhLTRiMDMtOTIxZC00NzY1NmJmNzA0MWMiLCJ0eXBlIjoicmVmcmVzaCIsImlhdCI6MTc2MzU3MzcwNiwiZXhwIjoxNzY0MTc4NTA2fQ.bZWsQjlXNQQkR7OG9GJ7UeJDbY7xOx8UA8u2eQq7uh0	2025-11-27 00:35:06.631257+07	f
53b9b6f5-609e-43bc-850c-78794b5ae883	70def630-fdd8-476e-8103-d2c53b2990ca	eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI3MGRlZjYzMC1mZGQ4LTQ3NmUtODEwMy1kMmM1M2IyOTkwY2EiLCJqdGkiOiI1M2I5YjZmNS02MDllLTQzYmMtODUwYy03ODc5NGI1YWU4ODMiLCJ0eXBlIjoicmVmcmVzaCIsImlhdCI6MTc2MzU3MzcwNiwiZXhwIjoxNzY0MTc4NTA2fQ.j6lAmtA3MMvHifahMZ2rXlWDYOirH2u3CDsPYBeadxo	2025-11-27 00:35:06.941766+07	f
ec743db8-2c7a-48dd-8c03-fa4dc8d2da04	f6439ac7-d46a-49f6-8dd5-fe98ed975726	eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJmNjQzOWFjNy1kNDZhLTQ5ZjYtOGRkNS1mZTk4ZWQ5NzU3MjYiLCJqdGkiOiJlYzc0M2RiOC0yYzdhLTQ4ZGQtOGMwMy1mYTRkYzhkMmRhMDQiLCJ0eXBlIjoicmVmcmVzaCIsImlhdCI6MTc2MzU3NjQ2NywiZXhwIjoxNzY0MTgxMjY3fQ.iDjz7QyuSDj0PSwIAWIMFcLOyZjKxx6MkCD80e2fFu4	2025-11-27 01:21:07.472483+07	f
2b81cea9-548f-419f-b11b-abeaceb3a56b	f6439ac7-d46a-49f6-8dd5-fe98ed975726	eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJmNjQzOWFjNy1kNDZhLTQ5ZjYtOGRkNS1mZTk4ZWQ5NzU3MjYiLCJqdGkiOiIyYjgxY2VhOS01NDhmLTQxOWYtYjExYi1hYmVhY2ViM2E1NmIiLCJ0eXBlIjoicmVmcmVzaCIsImlhdCI6MTc2MzU3NjQ5NSwiZXhwIjoxNzY0MTgxMjk1fQ.0xkQBi8SCrIleuIOTvFEu_0EqOKQXFL0ZOXQHqj1U3k	2025-11-27 01:21:35.28668+07	f
f45173b9-eabe-400a-8833-468cfa1c07a5	f6439ac7-d46a-49f6-8dd5-fe98ed975726	eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJmNjQzOWFjNy1kNDZhLTQ5ZjYtOGRkNS1mZTk4ZWQ5NzU3MjYiLCJqdGkiOiJmNDUxNzNiOS1lYWJlLTQwMGEtODgzMy00NjhjZmExYzA3YTUiLCJ0eXBlIjoicmVmcmVzaCIsImlhdCI6MTc2MzU3Njk3NCwiZXhwIjoxNzY0MTgxNzc0fQ.AEDfOHUEiiMDyJJQOZ-3ZMJ5oYULQ9gZX19JePnBQPI	2025-11-27 01:29:34.787723+07	f
52d9f505-62b0-4148-b747-770b115ed731	dbbe6c81-4380-4021-9a87-99df88e22f0b	eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJkYmJlNmM4MS00MzgwLTQwMjEtOWE4Ny05OWRmODhlMjJmMGIiLCJqdGkiOiI1MmQ5ZjUwNS02MmIwLTQxNDgtYjc0Ny03NzBiMTE1ZWQ3MzEiLCJ0eXBlIjoicmVmcmVzaCIsImlhdCI6MTc2MzU3NzUyMiwiZXhwIjoxNzY0MTgyMzIyfQ.m1thbFVveCPT0uSyRqYTuC5Nz965-NJJCngq0sCwB9g	2025-11-27 01:38:42.702079+07	f
35143a63-51a7-4399-8f95-4384b9266706	4ace5bbc-4f89-4a61-a936-36d24271af2b	eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI0YWNlNWJiYy00Zjg5LTRhNjEtYTkzNi0zNmQyNDI3MWFmMmIiLCJqdGkiOiIzNTE0M2E2My01MWE3LTQzOTktOGY5NS00Mzg0YjkyNjY3MDYiLCJ0eXBlIjoicmVmcmVzaCIsImlhdCI6MTc2MzU3OTkxNiwiZXhwIjoxNzY0MTg0NzE2fQ.NLMRdXctT4XVHWFRkSYu6JeEQZMsAP22HhAOzl5D3vc	2025-11-27 02:18:36.239721+07	f
4e4b5fcb-760f-4d07-9dfd-33c81425bc83	4ace5bbc-4f89-4a61-a936-36d24271af2b	eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI0YWNlNWJiYy00Zjg5LTRhNjEtYTkzNi0zNmQyNDI3MWFmMmIiLCJqdGkiOiI0ZTRiNWZjYi03NjBmLTRkMDctOWRmZC0zM2M4MTQyNWJjODMiLCJ0eXBlIjoicmVmcmVzaCIsImlhdCI6MTc2MzU3OTkyNiwiZXhwIjoxNzY0MTg0NzI2fQ.pe5PbXRDDLaZq_9_RJ3GpNMlMUrMDI_K3rVnIny7x28	2025-11-27 02:18:46.686616+07	f
01506025-017f-4596-9544-93495436c89f	4ace5bbc-4f89-4a61-a936-36d24271af2b	eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI0YWNlNWJiYy00Zjg5LTRhNjEtYTkzNi0zNmQyNDI3MWFmMmIiLCJqdGkiOiIwMTUwNjAyNS0wMTdmLTQ1OTYtOTU0NC05MzQ5NTQzNmM4OWYiLCJ0eXBlIjoicmVmcmVzaCIsImlhdCI6MTc2MzU4MjkxNiwiZXhwIjoxNzY0MTg3NzE2fQ.5yaiTPs7IMGu-WwLnEovx_yP34JWoswP0NBE4JpCuHo	2025-11-27 03:08:36.973187+07	f
7f748cdc-9b44-4e5f-abfe-95ebdb52bee7	4ace5bbc-4f89-4a61-a936-36d24271af2b	eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI0YWNlNWJiYy00Zjg5LTRhNjEtYTkzNi0zNmQyNDI3MWFmMmIiLCJqdGkiOiI3Zjc0OGNkYy05YjQ0LTRlNWYtYWJmZS05NWViZGI1MmJlZTciLCJ0eXBlIjoicmVmcmVzaCIsImlhdCI6MTc2MzU4MjkyNCwiZXhwIjoxNzY0MTg3NzI0fQ.9_c4HykbUB-ycsBd5-nxZ5IWJ43O8JSZVFdC6j0qfVc	2025-11-27 03:08:44.682416+07	f
c43aebb7-a992-4623-a68c-a8d640ffd691	4ace5bbc-4f89-4a61-a936-36d24271af2b	eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI0YWNlNWJiYy00Zjg5LTRhNjEtYTkzNi0zNmQyNDI3MWFmMmIiLCJqdGkiOiJjNDNhZWJiNy1hOTkyLTQ2MjMtYTY4Yy1hOGQ2NDBmZmQ2OTEiLCJ0eXBlIjoicmVmcmVzaCIsImlhdCI6MTc2MzU4MjkzMCwiZXhwIjoxNzY0MTg3NzMwfQ.jvNRZ1ecl2Uf305YKM3D2zrwwJdNQwQm__b__tLdHHo	2025-11-27 03:08:50.627666+07	f
405b0bb5-5004-4629-88d0-6fa29d644cbb	4ace5bbc-4f89-4a61-a936-36d24271af2b	eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI0YWNlNWJiYy00Zjg5LTRhNjEtYTkzNi0zNmQyNDI3MWFmMmIiLCJqdGkiOiI0MDViMGJiNS01MDA0LTQ2MjktODhkMC02ZmEyOWQ2NDRjYmIiLCJ0eXBlIjoicmVmcmVzaCIsImlhdCI6MTc2MzU4MzM3NCwiZXhwIjoxNzY0MTg4MTc0fQ.GS7Noon9VPkeBuN1AjU2-kl4qDFlySKuWkpKyP1dZPM	2025-11-27 03:16:14.722608+07	f
f56dc5c4-5067-4c1b-9591-45a65c12d5fa	4ace5bbc-4f89-4a61-a936-36d24271af2b	eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI0YWNlNWJiYy00Zjg5LTRhNjEtYTkzNi0zNmQyNDI3MWFmMmIiLCJqdGkiOiJmNTZkYzVjNC01MDY3LTRjMWItOTU5MS00NWE2NWMxMmQ1ZmEiLCJ0eXBlIjoicmVmcmVzaCIsImlhdCI6MTc2MzU4Mzg1MSwiZXhwIjoxNzY0MTg4NjUxfQ.sYKWqtN3EM0Osv-16jsL-NWTVHROfJHFiUjmE5IR6Z0	2025-11-27 03:24:11.559556+07	f
a326761c-3d49-4da4-b070-891cb0d44a78	4ace5bbc-4f89-4a61-a936-36d24271af2b	eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI0YWNlNWJiYy00Zjg5LTRhNjEtYTkzNi0zNmQyNDI3MWFmMmIiLCJqdGkiOiJhMzI2NzYxYy0zZDQ5LTRkYTQtYjA3MC04OTFjYjBkNDRhNzgiLCJ0eXBlIjoicmVmcmVzaCIsImlhdCI6MTc2MzU4NDI1NCwiZXhwIjoxNzY0MTg5MDU0fQ.BhrN9saCJptB2lrSC5WavJne3JPdVNxJfNpUlypCCMI	2025-11-27 03:30:54.220296+07	f
38071950-b79e-4de6-9bc5-010895e1018a	4ace5bbc-4f89-4a61-a936-36d24271af2b	eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI0YWNlNWJiYy00Zjg5LTRhNjEtYTkzNi0zNmQyNDI3MWFmMmIiLCJqdGkiOiIzODA3MTk1MC1iNzllLTRkZTYtOWJjNS0wMTA4OTVlMTAxOGEiLCJ0eXBlIjoicmVmcmVzaCIsImlhdCI6MTc2MzU4NTUwNywiZXhwIjoxNzY0MTkwMzA3fQ.cjPOSnPC-aylp5f8bVQWquMw5cYGqJm1UCS7z4gn-N8	2025-11-27 03:51:47.067344+07	f
be00e6d0-56bd-4a96-b847-b8b4cac41c1f	4ace5bbc-4f89-4a61-a936-36d24271af2b	eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI0YWNlNWJiYy00Zjg5LTRhNjEtYTkzNi0zNmQyNDI3MWFmMmIiLCJqdGkiOiJiZTAwZTZkMC01NmJkLTRhOTYtYjg0Ny1iOGI0Y2FjNDFjMWYiLCJ0eXBlIjoicmVmcmVzaCIsImlhdCI6MTc2MzU4Njc4OCwiZXhwIjoxNzY0MTkxNTg4fQ.tZcFXgqWLL0_Me6ckrFcdBIhricSb2UfFVL4IkFry6E	2025-11-27 04:13:08.224878+07	f
0ff0d2a3-d982-4a7c-8e7e-bbf8b1d35ccf	4ace5bbc-4f89-4a61-a936-36d24271af2b	eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI0YWNlNWJiYy00Zjg5LTRhNjEtYTkzNi0zNmQyNDI3MWFmMmIiLCJqdGkiOiIwZmYwZDJhMy1kOTgyLTRhN2MtOGU3ZS1iYmY4YjFkMzVjY2YiLCJ0eXBlIjoicmVmcmVzaCIsImlhdCI6MTc2MzU4ODE4OCwiZXhwIjoxNzY0MTkyOTg4fQ.ytgrtqPGm6ErCMg1yqOt--GQgd7qAuknracv5pXCGG4	2025-11-27 04:36:28.959098+07	f
b39bdc15-62a8-4360-bedc-c5cc6cc16c8f	4ace5bbc-4f89-4a61-a936-36d24271af2b	eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI0YWNlNWJiYy00Zjg5LTRhNjEtYTkzNi0zNmQyNDI3MWFmMmIiLCJqdGkiOiJiMzliZGMxNS02MmE4LTQzNjAtYmVkYy1jNWNjNmNjMTZjOGYiLCJ0eXBlIjoicmVmcmVzaCIsImlhdCI6MTc2MzU4OTMyOSwiZXhwIjoxNzY0MTk0MTI5fQ.DwSot0_xR4aaLYLk180FlyEDavUdGb09HCoI17HXAXA	2025-11-27 04:55:29.500567+07	f
48c6c70e-0dec-425f-a9a2-fb42f33ace73	4ace5bbc-4f89-4a61-a936-36d24271af2b	eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI0YWNlNWJiYy00Zjg5LTRhNjEtYTkzNi0zNmQyNDI3MWFmMmIiLCJqdGkiOiI0OGM2YzcwZS0wZGVjLTQyNWYtYTlhMi1mYjQyZjMzYWNlNzMiLCJ0eXBlIjoicmVmcmVzaCIsImlhdCI6MTc2MzU5MDU0NCwiZXhwIjoxNzY0MTk1MzQ0fQ.c0APcZe6Zk8P8etn_qPg1A49pgOeMFUbmeIhxFtEZiw	2025-11-27 05:15:44.609261+07	f
a9c27e90-f938-4cfe-85bd-0d4ba5f8a9af	4ace5bbc-4f89-4a61-a936-36d24271af2b	eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI0YWNlNWJiYy00Zjg5LTRhNjEtYTkzNi0zNmQyNDI3MWFmMmIiLCJqdGkiOiJhOWMyN2U5MC1mOTM4LTRjZmUtODViZC0wZDRiYTVmOGE5YWYiLCJ0eXBlIjoicmVmcmVzaCIsImlhdCI6MTc2MzU5MTA2MSwiZXhwIjoxNzY0MTk1ODYxfQ.dFf0ANPuP4bxIeYD1NVf-NrrjdbvxLPKbYz27QYBuT4	2025-11-27 05:24:21.301507+07	f
d06beac4-3b0f-4682-9da5-40d17a43035d	4ace5bbc-4f89-4a61-a936-36d24271af2b	eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI0YWNlNWJiYy00Zjg5LTRhNjEtYTkzNi0zNmQyNDI3MWFmMmIiLCJqdGkiOiJkMDZiZWFjNC0zYjBmLTQ2ODItOWRhNS00MGQxN2E0MzAzNWQiLCJ0eXBlIjoicmVmcmVzaCIsImlhdCI6MTc2MzU5Mjg1MCwiZXhwIjoxNzY0MTk3NjUwfQ.T-FK96QDBXFfd5lkT3vEe1uZ09MtcB5JqTic-nVy7k0	2025-11-27 05:54:10.700714+07	f
320481e8-6a03-40eb-bb86-d23dc6575de1	1bebabf4-19b5-43d1-8677-c0dcd2bcc928	eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxYmViYWJmNC0xOWI1LTQzZDEtODY3Ny1jMGRjZDJiY2M5MjgiLCJqdGkiOiIzMjA0ODFlOC02YTAzLTQwZWItYmI4Ni1kMjNkYzY1NzVkZTEiLCJ0eXBlIjoicmVmcmVzaCIsImlhdCI6MTc2MzU5MzkwMiwiZXhwIjoxNzY0MTk4NzAyfQ.VUkjX-DvEOIh5a9jcPDEK7rgmFu9wHNKYwAJJxoptw8	2025-11-27 06:11:42.180018+07	f
fba19b9d-ce4e-4462-b15f-d2d5a36ad899	1bebabf4-19b5-43d1-8677-c0dcd2bcc928	eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxYmViYWJmNC0xOWI1LTQzZDEtODY3Ny1jMGRjZDJiY2M5MjgiLCJqdGkiOiJmYmExOWI5ZC1jZTRlLTQ0NjItYjE1Zi1kMmQ1YTM2YWQ4OTkiLCJ0eXBlIjoicmVmcmVzaCIsImlhdCI6MTc2MzU5MzkxNSwiZXhwIjoxNzY0MTk4NzE1fQ.cSIR2HGqDfYM92sry8TkDctuE5Mbr9QrqQL_XMTCqEQ	2025-11-27 06:11:55.775706+07	f
7ec2fbdd-0fc0-4212-b842-d1b630e8b16e	1bebabf4-19b5-43d1-8677-c0dcd2bcc928	eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxYmViYWJmNC0xOWI1LTQzZDEtODY3Ny1jMGRjZDJiY2M5MjgiLCJqdGkiOiI3ZWMyZmJkZC0wZmMwLTQyMTItYjg0Mi1kMWI2MzBlOGIxNmUiLCJ0eXBlIjoicmVmcmVzaCIsImlhdCI6MTc2MzU5NDkxNywiZXhwIjoxNzY0MTk5NzE3fQ.aGjFKFifG-dqP9RmXk9Ea2aollIPTfYTye5KFYgum2I	2025-11-27 06:28:37.915889+07	f
53efa239-789e-450c-af03-cac1548bf63f	4ace5bbc-4f89-4a61-a936-36d24271af2b	eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI0YWNlNWJiYy00Zjg5LTRhNjEtYTkzNi0zNmQyNDI3MWFmMmIiLCJqdGkiOiI1M2VmYTIzOS03ODllLTQ1MGMtYWYwMy1jYWMxNTQ4YmY2M2YiLCJ0eXBlIjoicmVmcmVzaCIsImlhdCI6MTc2MzYzODYxNiwiZXhwIjoxNzY0MjQzNDE2fQ.dXgA-3iXtNPZ3-VZNTIaSSrReu2831eeAqPfLXTFT2U	2025-11-27 18:36:56.542849+07	f
2e36efde-0c2a-44f7-a9d4-6891afba11cb	4ace5bbc-4f89-4a61-a936-36d24271af2b	eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI0YWNlNWJiYy00Zjg5LTRhNjEtYTkzNi0zNmQyNDI3MWFmMmIiLCJqdGkiOiIyZTM2ZWZkZS0wYzJhLTQ0ZjctYTlkNC02ODkxYWZiYTExY2IiLCJ0eXBlIjoicmVmcmVzaCIsImlhdCI6MTc2MzYzOTg5OCwiZXhwIjoxNzY0MjQ0Njk4fQ.oxeubc80erF0f9qnjjM2I0YIWFqQAUugT-g42RWsxCw	2025-11-27 18:58:18.479344+07	f
54387ce5-9102-4482-a533-3c82f69d349b	4ace5bbc-4f89-4a61-a936-36d24271af2b	eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI0YWNlNWJiYy00Zjg5LTRhNjEtYTkzNi0zNmQyNDI3MWFmMmIiLCJqdGkiOiI1NDM4N2NlNS05MTAyLTQ0ODItYTUzMy0zYzgyZjY5ZDM0OWIiLCJ0eXBlIjoicmVmcmVzaCIsImlhdCI6MTc2MzY0MDk1MywiZXhwIjoxNzY0MjQ1NzUzfQ.DW3N6wqywO0JOal1JtowVQQqPNXs13MZyauGM0YjEaE	2025-11-27 19:15:53.748098+07	f
71a77555-9346-4aa1-9e74-fb87faa437ec	4ace5bbc-4f89-4a61-a936-36d24271af2b	eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI0YWNlNWJiYy00Zjg5LTRhNjEtYTkzNi0zNmQyNDI3MWFmMmIiLCJqdGkiOiI3MWE3NzU1NS05MzQ2LTRhYTEtOWU3NC1mYjg3ZmFhNDM3ZWMiLCJ0eXBlIjoicmVmcmVzaCIsImlhdCI6MTc2MzY0Mjc3OSwiZXhwIjoxNzY0MjQ3NTc5fQ.XgkYOQP9wONSrrVBXZ9_nIoE6bVSioKc0u_8e-0FwvI	2025-11-27 19:46:19.813875+07	f
1f368e2f-1d18-44ac-837f-c9f38f3166c1	4ace5bbc-4f89-4a61-a936-36d24271af2b	eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI0YWNlNWJiYy00Zjg5LTRhNjEtYTkzNi0zNmQyNDI3MWFmMmIiLCJqdGkiOiIxZjM2OGUyZi0xZDE4LTQ0YWMtODM3Zi1jOWYzOGYzMTY2YzEiLCJ0eXBlIjoicmVmcmVzaCIsImlhdCI6MTc2MzY1NTE2NCwiZXhwIjoxNzY0MjU5OTY0fQ.zgR_fYcTONSP6i8rVuMcb95CtJN4VHoyGZ0p03ql5Z0	2025-11-27 23:12:44.205887+07	f
3e00145f-6e6f-4944-8c5d-b8e6c5a8c8a4	5252be7a-5169-4316-8802-c4500826907d	eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI1MjUyYmU3YS01MTY5LTQzMTYtODgwMi1jNDUwMDgyNjkwN2QiLCJqdGkiOiIzZTAwMTQ1Zi02ZTZmLTQ5NDQtOGM1ZC1iOGU2YzVhOGM4YTQiLCJ0eXBlIjoicmVmcmVzaCIsImlhdCI6MTc2MzY1Njc2NywiZXhwIjoxNzY0MjYxNTY3fQ.apRW2yK0kYmRpGIuj8xZ6-RbVybFZErvgOxEEFQ_EwU	2025-11-27 23:39:27.171828+07	f
105c588b-899a-4d88-b8bb-c842af6d56be	6bcc2e04-e0c7-48cc-ab33-a48f4f4e643d	eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI2YmNjMmUwNC1lMGM3LTQ4Y2MtYWIzMy1hNDhmNGY0ZTY0M2QiLCJqdGkiOiIxMDVjNTg4Yi04OTlhLTRkODgtYjhiYi1jODQyYWY2ZDU2YmUiLCJ0eXBlIjoicmVmcmVzaCIsImlhdCI6MTc2MzY1ODE3NywiZXhwIjoxNzY0MjYyOTc3fQ.kSFzJIB9Y1IKW6UWTCDjhNfXdSxlL1IEAjm1t0YeHKc	2025-11-28 00:02:57.547049+07	f
541c3cb3-b2d3-4ddb-bbda-f13b80c27bd0	4ace5bbc-4f89-4a61-a936-36d24271af2b	eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI0YWNlNWJiYy00Zjg5LTRhNjEtYTkzNi0zNmQyNDI3MWFmMmIiLCJqdGkiOiI1NDFjM2NiMy1iMmQzLTRkZGItYmJkYS1mMTNiODBjMjdiZDAiLCJ0eXBlIjoicmVmcmVzaCIsImlhdCI6MTc2MzY1ODE4OSwiZXhwIjoxNzY0MjYyOTg5fQ.hY_9A6FplMKpbmqW4B7vKCk4cD_oAP68pn4Qj9NNYGk	2025-11-28 00:03:09.962699+07	f
70312acc-6bb4-4b67-9fde-d305b08e84ac	4ace5bbc-4f89-4a61-a936-36d24271af2b	eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI0YWNlNWJiYy00Zjg5LTRhNjEtYTkzNi0zNmQyNDI3MWFmMmIiLCJqdGkiOiI3MDMxMmFjYy02YmI0LTRiNjctOWZkZS1kMzA1YjA4ZTg0YWMiLCJ0eXBlIjoicmVmcmVzaCIsImlhdCI6MTc2MzY2MTkwMSwiZXhwIjoxNzY0MjY2NzAxfQ.Xy8j7XSrAMLVwlOBEbouYRKNxHEZv9Tm42sO2MMPY04	2025-11-28 01:05:01.242335+07	f
cadbb79a-b184-4437-bef6-6a4710dd5734	4ace5bbc-4f89-4a61-a936-36d24271af2b	eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI0YWNlNWJiYy00Zjg5LTRhNjEtYTkzNi0zNmQyNDI3MWFmMmIiLCJqdGkiOiJjYWRiYjc5YS1iMTg0LTQ0MzctYmVmNi02YTQ3MTBkZDU3MzQiLCJ0eXBlIjoicmVmcmVzaCIsImlhdCI6MTc2MzY2MzAyMywiZXhwIjoxNzY0MjY3ODIzfQ.LSgB1Wmd5vQnIx8xrNG4QrsOEZApFXzT9iTKamYF3AM	2025-11-28 01:23:43.848851+07	f
ad0925b3-022a-4692-8d65-58b437393607	4ace5bbc-4f89-4a61-a936-36d24271af2b	eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI0YWNlNWJiYy00Zjg5LTRhNjEtYTkzNi0zNmQyNDI3MWFmMmIiLCJqdGkiOiJhZDA5MjViMy0wMjJhLTQ2OTItOGQ2NS01OGI0MzczOTM2MDciLCJ0eXBlIjoicmVmcmVzaCIsImlhdCI6MTc2MzY2NDA5OSwiZXhwIjoxNzY0MjY4ODk5fQ._aOC86SnUxIcMOkP7VRmepAw844qUWZ5RLFi5jeOmmg	2025-11-28 01:41:39.792962+07	f
47a19295-7e10-4d1a-8ca0-93b34b6ad620	4ace5bbc-4f89-4a61-a936-36d24271af2b	eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI0YWNlNWJiYy00Zjg5LTRhNjEtYTkzNi0zNmQyNDI3MWFmMmIiLCJqdGkiOiI0N2ExOTI5NS03ZTEwLTRkMWEtOGNhMC05M2IzNGI2YWQ2MjAiLCJ0eXBlIjoicmVmcmVzaCIsImlhdCI6MTc2MzY2NTIwMSwiZXhwIjoxNzY0MjcwMDAxfQ.Vy5Et3_HCBLfIuB8o2GjqORfvppFJAQ_QgPCf3FLs90	2025-11-28 02:00:01.099135+07	f
0a3aa391-c0fd-4c6c-b204-1bc31edeefb8	4ace5bbc-4f89-4a61-a936-36d24271af2b	eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI0YWNlNWJiYy00Zjg5LTRhNjEtYTkzNi0zNmQyNDI3MWFmMmIiLCJqdGkiOiIwYTNhYTM5MS1jMGZkLTRjNmMtYjIwNC0xYmMzMWVkZWVmYjgiLCJ0eXBlIjoicmVmcmVzaCIsImlhdCI6MTc2MzY2NjM4OCwiZXhwIjoxNzY0MjcxMTg4fQ.zMtZLD_NyKxNEnDAKGE1VmUb8PT9sQ7_YEOqNOrjRQY	2025-11-28 02:19:48.471638+07	f
cabd50a6-4133-44f6-b3d1-27266b47111c	4ace5bbc-4f89-4a61-a936-36d24271af2b	eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI0YWNlNWJiYy00Zjg5LTRhNjEtYTkzNi0zNmQyNDI3MWFmMmIiLCJqdGkiOiJjYWJkNTBhNi00MTMzLTQ0ZjYtYjNkMS0yNzI2NmI0NzExMWMiLCJ0eXBlIjoicmVmcmVzaCIsImlhdCI6MTc2MzY2NzUyNCwiZXhwIjoxNzY0MjcyMzI0fQ.TGCVWtCrLu6FmqE2WoyIwoRzUE1GHbCw4I4XZpSmPzw	2025-11-28 02:38:44.20386+07	f
2f8884df-49b9-40a4-8404-3525b1378dc1	4ace5bbc-4f89-4a61-a936-36d24271af2b	eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI0YWNlNWJiYy00Zjg5LTRhNjEtYTkzNi0zNmQyNDI3MWFmMmIiLCJqdGkiOiIyZjg4ODRkZi00OWI5LTQwYTQtODQwNC0zNTI1YjEzNzhkYzEiLCJ0eXBlIjoicmVmcmVzaCIsImlhdCI6MTc2MzY2OTAxMCwiZXhwIjoxNzY0MjczODEwfQ.cfEV5DiZHzvZnX_9l2GYrH6wTFiyCY2Gs8awh5TsrhY	2025-11-28 03:03:30.246133+07	f
099b90ab-3482-4f8d-acf1-a3bd1d5b22ae	4ace5bbc-4f89-4a61-a936-36d24271af2b	eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI0YWNlNWJiYy00Zjg5LTRhNjEtYTkzNi0zNmQyNDI3MWFmMmIiLCJqdGkiOiIwOTliOTBhYi0zNDgyLTRmOGQtYWNmMS1hM2JkMWQ1YjIyYWUiLCJ0eXBlIjoicmVmcmVzaCIsImlhdCI6MTc2MzY3MDY2MSwiZXhwIjoxNzY0Mjc1NDYxfQ.NFpPWwdkohkqttxGrgOopuLHxF6SzfVu5CXPb14rjTY	2025-11-28 03:31:01.109829+07	f
94667082-f953-4a11-b192-c2e6857ff494	4ace5bbc-4f89-4a61-a936-36d24271af2b	eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI0YWNlNWJiYy00Zjg5LTRhNjEtYTkzNi0zNmQyNDI3MWFmMmIiLCJqdGkiOiI5NDY2NzA4Mi1mOTUzLTRhMTEtYjE5Mi1jMmU2ODU3ZmY0OTQiLCJ0eXBlIjoicmVmcmVzaCIsImlhdCI6MTc2MzY3MTcxMywiZXhwIjoxNzY0Mjc2NTEzfQ.5IFxFTqQBCZ5WFeLEtfG1N_C1LQQTYJwJLFj_Y9wj-U	2025-11-28 03:48:33.728551+07	f
0b1de741-cec8-4865-90a7-ca40a9efcc11	4ace5bbc-4f89-4a61-a936-36d24271af2b	eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI0YWNlNWJiYy00Zjg5LTRhNjEtYTkzNi0zNmQyNDI3MWFmMmIiLCJqdGkiOiIwYjFkZTc0MS1jZWM4LTQ4NjUtOTBhNy1jYTQwYTllZmNjMTEiLCJ0eXBlIjoicmVmcmVzaCIsImlhdCI6MTc2MzY3MzIyMSwiZXhwIjoxNzY0Mjc4MDIxfQ.XL9mN189244CRHVWE6qTbyUTt0SLU32Iz4t1N6RcBbo	2025-11-28 04:13:41.706468+07	f
8f865f0c-cdb8-4d3a-919e-a6b4317fcb50	4ace5bbc-4f89-4a61-a936-36d24271af2b	eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI0YWNlNWJiYy00Zjg5LTRhNjEtYTkzNi0zNmQyNDI3MWFmMmIiLCJqdGkiOiI4Zjg2NWYwYy1jZGI4LTRkM2EtOTE5ZS1hNmI0MzE3ZmNiNTAiLCJ0eXBlIjoicmVmcmVzaCIsImlhdCI6MTc2MzY3NTc4MiwiZXhwIjoxNzY0MjgwNTgyfQ.diyylekaMYsHeDfSL7d-tgBLgKQckW9lNsgFbpssTfw	2025-11-28 04:56:22.946095+07	f
d656df47-d886-4ea4-ac51-020a8db64ef5	4ace5bbc-4f89-4a61-a936-36d24271af2b	eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI0YWNlNWJiYy00Zjg5LTRhNjEtYTkzNi0zNmQyNDI3MWFmMmIiLCJqdGkiOiJkNjU2ZGY0Ny1kODg2LTRlYTQtYWM1MS0wMjBhOGRiNjRlZjUiLCJ0eXBlIjoicmVmcmVzaCIsImlhdCI6MTc2MzY3Njc0NSwiZXhwIjoxNzY0MjgxNTQ1fQ.Fw5pl-hlLr2vG_5Gz31G5gc2CEK5u6yKtSjH4r7GoTk	2025-11-28 05:12:25.561551+07	f
1ac5a6d6-bfd9-4fc8-8326-df839e95ca4e	4ace5bbc-4f89-4a61-a936-36d24271af2b	eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI0YWNlNWJiYy00Zjg5LTRhNjEtYTkzNi0zNmQyNDI3MWFmMmIiLCJqdGkiOiIxYWM1YTZkNi1iZmQ5LTRmYzgtODMyNi1kZjgzOWU5NWNhNGUiLCJ0eXBlIjoicmVmcmVzaCIsImlhdCI6MTc2MzY3ODMxNiwiZXhwIjoxNzY0MjgzMTE2fQ.The-1XL-iPYXQ1g-x3JPmZvztig4x7wdSpyADLdc9UA	2025-11-28 05:38:36.151457+07	f
caff60af-1429-48dd-9916-623d853753bf	4ace5bbc-4f89-4a61-a936-36d24271af2b	eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI0YWNlNWJiYy00Zjg5LTRhNjEtYTkzNi0zNmQyNDI3MWFmMmIiLCJqdGkiOiJjYWZmNjBhZi0xNDI5LTQ4ZGQtOTkxNi02MjNkODUzNzUzYmYiLCJ0eXBlIjoicmVmcmVzaCIsImlhdCI6MTc2MzY3OTM0MiwiZXhwIjoxNzY0Mjg0MTQyfQ.-Xw_2ZmvndPA_CfP1MSlziSRn6YRJfg1fAyYs6ufYl8	2025-11-28 05:55:42.064784+07	f
c7ffdbd6-fcb7-41ec-acbf-28ac96aeed50	4ace5bbc-4f89-4a61-a936-36d24271af2b	eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI0YWNlNWJiYy00Zjg5LTRhNjEtYTkzNi0zNmQyNDI3MWFmMmIiLCJqdGkiOiJjN2ZmZGJkNi1mY2I3LTQxZWMtYWNiZi0yOGFjOTZhZWVkNTAiLCJ0eXBlIjoicmVmcmVzaCIsImlhdCI6MTc2MzY4MDI4NCwiZXhwIjoxNzY0Mjg1MDg0fQ.EDLgfpyWgsItKsedIum0lNJax8haAv1KDp9PR9VBHBw	2025-11-28 06:11:24.25998+07	f
6ae5e064-8ffa-4409-909e-27c0b449ec05	1bebabf4-19b5-43d1-8677-c0dcd2bcc928	eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxYmViYWJmNC0xOWI1LTQzZDEtODY3Ny1jMGRjZDJiY2M5MjgiLCJqdGkiOiI2YWU1ZTA2NC04ZmZhLTQ0MDktOTA5ZS0yN2MwYjQ0OWVjMDUiLCJ0eXBlIjoicmVmcmVzaCIsImlhdCI6MTc2MzY4MTYyNSwiZXhwIjoxNzY0Mjg2NDI1fQ.YE30zaci9Kbw-qKDJsj8WV40ituziVi62VCzPatQ4Q8	2025-11-28 06:33:45.607256+07	f
43e2ac46-3fde-4bb4-8719-8b9889831f65	4ace5bbc-4f89-4a61-a936-36d24271af2b	eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI0YWNlNWJiYy00Zjg5LTRhNjEtYTkzNi0zNmQyNDI3MWFmMmIiLCJqdGkiOiI0M2UyYWM0Ni0zZmRlLTRiYjQtODcxOS04Yjk4ODk4MzFmNjUiLCJ0eXBlIjoicmVmcmVzaCIsImlhdCI6MTc2MzY4MTcyOSwiZXhwIjoxNzY0Mjg2NTI5fQ.IPY2twVvY5zvoUM3zbTNNe1v_faN2OhtuYDXr4P2GME	2025-11-28 06:35:29.931836+07	f
c9c87c78-dfb2-40dd-9b7f-b5ed1f25b029	4ace5bbc-4f89-4a61-a936-36d24271af2b	eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI0YWNlNWJiYy00Zjg5LTRhNjEtYTkzNi0zNmQyNDI3MWFmMmIiLCJqdGkiOiJjOWM4N2M3OC1kZmIyLTQwZGQtOWI3Zi1iNWVkMWYyNWIwMjkiLCJ0eXBlIjoicmVmcmVzaCIsImlhdCI6MTc2MzY4MTg1MCwiZXhwIjoxNzY0Mjg2NjUwfQ.OP0hlqheiusw_pHn1dsgGFHPtHJPPwibFRPF_GLy1Ks	2025-11-28 06:37:30.359536+07	f
3c211ee4-4a72-4d78-95d0-78318ddff7d0	4ace5bbc-4f89-4a61-a936-36d24271af2b	eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI0YWNlNWJiYy00Zjg5LTRhNjEtYTkzNi0zNmQyNDI3MWFmMmIiLCJqdGkiOiIzYzIxMWVlNC00YTcyLTRkNzgtOTVkMC03ODMxOGRkZmY3ZDAiLCJ0eXBlIjoicmVmcmVzaCIsImlhdCI6MTc2MzY4MjEzNywiZXhwIjoxNzY0Mjg2OTM3fQ.lxm8xJ1UCfwawBh6UtUVUpPo3ZF2kdwCJXUqcZg8zKM	2025-11-28 06:42:17.454822+07	f
e14dff4e-6a8d-4eda-96c4-cf9dc6107084	4ace5bbc-4f89-4a61-a936-36d24271af2b	eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI0YWNlNWJiYy00Zjg5LTRhNjEtYTkzNi0zNmQyNDI3MWFmMmIiLCJqdGkiOiJlMTRkZmY0ZS02YThkLTRlZGEtOTZjNC1jZjlkYzYxMDcwODQiLCJ0eXBlIjoicmVmcmVzaCIsImlhdCI6MTc2MzY4MjY1NSwiZXhwIjoxNzY0Mjg3NDU1fQ.TXO3VcUSzRkU_l_7lD28aRYlOeA_HGykPQc6ea04ViY	2025-11-28 06:50:55.863931+07	f
496413dc-1ac7-41c0-bf49-bd1f24aa2cf6	4ace5bbc-4f89-4a61-a936-36d24271af2b	eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI0YWNlNWJiYy00Zjg5LTRhNjEtYTkzNi0zNmQyNDI3MWFmMmIiLCJqdGkiOiI0OTY0MTNkYy0xYWM3LTQxYzAtYmY0OS1iZDFmMjRhYTJjZjYiLCJ0eXBlIjoicmVmcmVzaCIsImlhdCI6MTc2MzY4MzU3MCwiZXhwIjoxNzY0Mjg4MzcwfQ.WksnLlRqTHBAp0vHe82U8rQlMEX7j8Ak8eBwQ9AIWPA	2025-11-28 07:06:10.907012+07	f
c9b234c3-e68e-4659-b0bd-f5a50a8bdb18	1bebabf4-19b5-43d1-8677-c0dcd2bcc928	eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxYmViYWJmNC0xOWI1LTQzZDEtODY3Ny1jMGRjZDJiY2M5MjgiLCJqdGkiOiJjOWIyMzRjMy1lNjhlLTQ2NTktYjBiZC1mNWE1MGE4YmRiMTgiLCJ0eXBlIjoicmVmcmVzaCIsImlhdCI6MTc2MzY4NDE3OSwiZXhwIjoxNzY0Mjg4OTc5fQ.l5Nkr9FjX_0cZeZdUE7MPRqvPOeLJohi-oJP05C-IdQ	2025-11-28 07:16:19.402148+07	f
778acf0b-bb07-4c3d-9ade-1144c2a70f5e	4ace5bbc-4f89-4a61-a936-36d24271af2b	eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI0YWNlNWJiYy00Zjg5LTRhNjEtYTkzNi0zNmQyNDI3MWFmMmIiLCJqdGkiOiI3NzhhY2YwYi1iYjA3LTRjM2QtOWFkZS0xMTQ0YzJhNzBmNWUiLCJ0eXBlIjoicmVmcmVzaCIsImlhdCI6MTc2MzcxNzM1MywiZXhwIjoxNzY0MzIyMTUzfQ.avP6MpWSDAiiBybKP1YZobEY2T7Fb9q0Z4fqXpVNJn8	2025-11-28 16:29:13.652611+07	f
0a3b4878-4a4c-4ac8-aa50-43af5bdb70b7	4ace5bbc-4f89-4a61-a936-36d24271af2b	eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI0YWNlNWJiYy00Zjg5LTRhNjEtYTkzNi0zNmQyNDI3MWFmMmIiLCJqdGkiOiIwYTNiNDg3OC00YTRjLTRhYzgtYWE1MC00M2FmNWJkYjcwYjciLCJ0eXBlIjoicmVmcmVzaCIsImlhdCI6MTc2MzcxODM1NiwiZXhwIjoxNzY0MzIzMTU2fQ._r5pBf0XG5Gr8yPL0oAs3ZVbwAeg5kFu8tZ8BlCtORA	2025-11-28 16:45:56.328247+07	f
659b5d8c-f19f-450c-9f42-37f10126944e	4ace5bbc-4f89-4a61-a936-36d24271af2b	eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI0YWNlNWJiYy00Zjg5LTRhNjEtYTkzNi0zNmQyNDI3MWFmMmIiLCJqdGkiOiI2NTliNWQ4Yy1mMTlmLTQ1MGMtOWY0Mi0zN2YxMDEyNjk0NGUiLCJ0eXBlIjoicmVmcmVzaCIsImlhdCI6MTc2Mzc0OTg5NSwiZXhwIjoxNzY0MzU0Njk1fQ.PsNVLW4yNq5mFI4FZduhFpClgwvhDHUftk6ymj5PXFQ	2025-11-29 01:31:35.868095+07	f
f1228a4f-e4a6-42f2-a874-a0dd97f545c3	4ace5bbc-4f89-4a61-a936-36d24271af2b	eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI0YWNlNWJiYy00Zjg5LTRhNjEtYTkzNi0zNmQyNDI3MWFmMmIiLCJqdGkiOiJmMTIyOGE0Zi1lNGE2LTQyZjItYTg3NC1hMGRkOTdmNTQ1YzMiLCJ0eXBlIjoicmVmcmVzaCIsImlhdCI6MTc2Mzc1MDgyMywiZXhwIjoxNzY0MzU1NjIzfQ.PEDteL_A9EXbuYoQjGnyi9nnla8igXivNGrmMCAce2I	2025-11-29 01:47:03.502804+07	f
f6dfb850-6310-409c-95ba-aa4142fbda11	4ace5bbc-4f89-4a61-a936-36d24271af2b	eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI0YWNlNWJiYy00Zjg5LTRhNjEtYTkzNi0zNmQyNDI3MWFmMmIiLCJqdGkiOiJmNmRmYjg1MC02MzEwLTQwOWMtOTViYS1hYTQxNDJmYmRhMTEiLCJ0eXBlIjoicmVmcmVzaCIsImlhdCI6MTc2Mzc1MjAwNCwiZXhwIjoxNzY0MzU2ODA0fQ.PEAKjxV6on81Hs76nbRi1p16FhcPv0abY4UC-ZxmwIE	2025-11-29 02:06:44.937433+07	f
150a392a-cd3d-4f68-8dab-2311aa0fa03b	4ace5bbc-4f89-4a61-a936-36d24271af2b	eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI0YWNlNWJiYy00Zjg5LTRhNjEtYTkzNi0zNmQyNDI3MWFmMmIiLCJqdGkiOiIxNTBhMzkyYS1jZDNkLTRmNjgtOGRhYi0yMzExYWEwZmEwM2IiLCJ0eXBlIjoicmVmcmVzaCIsImlhdCI6MTc2Mzc1MzAzMCwiZXhwIjoxNzY0MzU3ODMwfQ.W-JOUCkIHjj25Qm0Lnnh9IpDxT41v8zRFgZ_doS9Qwc	2025-11-29 02:23:50.555574+07	f
455c9b9f-cb6a-4fa1-9031-0706536c13cd	4ace5bbc-4f89-4a61-a936-36d24271af2b	eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI0YWNlNWJiYy00Zjg5LTRhNjEtYTkzNi0zNmQyNDI3MWFmMmIiLCJqdGkiOiI0NTVjOWI5Zi1jYjZhLTRmYTEtOTAzMS0wNzA2NTM2YzEzY2QiLCJ0eXBlIjoicmVmcmVzaCIsImlhdCI6MTc2Mzc1NDE4NCwiZXhwIjoxNzY0MzU4OTg0fQ.dTGwCjVSIhG9f8pwE5lO7--glhwGqSK58tmF_Ya7SRs	2025-11-29 02:43:04.503262+07	f
7bee6606-ae48-40d5-8d7b-426187656304	4ace5bbc-4f89-4a61-a936-36d24271af2b	eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI0YWNlNWJiYy00Zjg5LTRhNjEtYTkzNi0zNmQyNDI3MWFmMmIiLCJqdGkiOiI3YmVlNjYwNi1hZTQ4LTQwZDUtOGQ3Yi00MjYxODc2NTYzMDQiLCJ0eXBlIjoicmVmcmVzaCIsImlhdCI6MTc2Mzc1NTE5MiwiZXhwIjoxNzY0MzU5OTkyfQ.lilGprBjrftMusC9ntuc89hKGAAke4qUpGdFR4GQRWc	2025-11-29 02:59:52.089554+07	f
a5626792-2583-4642-9c5f-ffa09e74f4b6	8c76a289-3236-48c2-b681-10ca7747c638	eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI4Yzc2YTI4OS0zMjM2LTQ4YzItYjY4MS0xMGNhNzc0N2M2MzgiLCJqdGkiOiJhNTYyNjc5Mi0yNTgzLTQ2NDItOWM1Zi1mZmEwOWU3NGY0YjYiLCJ0eXBlIjoicmVmcmVzaCIsImlhdCI6MTc2MzkxMTA0MSwiZXhwIjoxNzY0NTE1ODQxfQ.wL0YyA32G0BOWMB0rEx7kVd_en9vCzF8iF2NDgdpMd8	2025-11-30 22:17:21.72029+07	f
db07860a-213e-4545-86ac-7664de73da01	9be85c2b-0b71-4234-b2e2-cb202499dab3	eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI5YmU4NWMyYi0wYjcxLTQyMzQtYjJlMi1jYjIwMjQ5OWRhYjMiLCJqdGkiOiJkYjA3ODYwYS0yMTNlLTQ1NDUtODZhYy03NjY0ZGU3M2RhMDEiLCJ0eXBlIjoicmVmcmVzaCIsImlhdCI6MTc2MzkxMTA4NywiZXhwIjoxNzY0NTE1ODg3fQ.T6cKoHDF2RPoyIXWBawrB08704Z8PXTcoKUVSUDmZjw	2025-11-30 22:18:07.976347+07	f
8c6a6158-604a-4787-a9d5-988f3a37918e	4ace5bbc-4f89-4a61-a936-36d24271af2b	eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI0YWNlNWJiYy00Zjg5LTRhNjEtYTkzNi0zNmQyNDI3MWFmMmIiLCJqdGkiOiI4YzZhNjE1OC02MDRhLTQ3ODctYTlkNS05ODhmM2EzNzkxOGUiLCJ0eXBlIjoicmVmcmVzaCIsImlhdCI6MTc2Mzc2NDcxOCwiZXhwIjoxNzY0MzY5NTE4fQ.24a7c4Ry0eMowJA8vzI3G0SIErhaMqOfqkzFE7GynSQ	2025-11-29 05:38:38.595107+07	f
e99f06c7-ef5d-4ebb-a689-6e9cb9e095d8	4ace5bbc-4f89-4a61-a936-36d24271af2b	eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI0YWNlNWJiYy00Zjg5LTRhNjEtYTkzNi0zNmQyNDI3MWFmMmIiLCJqdGkiOiJlOTlmMDZjNy1lZjVkLTRlYmItYTY4OS02ZTljYjllMDk1ZDgiLCJ0eXBlIjoicmVmcmVzaCIsImlhdCI6MTc2MzkwMDM5MSwiZXhwIjoxNzY0NTA1MTkxfQ.K-cdw7TZpYCjx74EAg2bbVq_1clfxIJrBEo0Ryf2XOU	2025-11-30 19:19:51.607576+07	f
d1ee4364-9259-436a-b643-8c7dd47f9f9c	4ace5bbc-4f89-4a61-a936-36d24271af2b	eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI0YWNlNWJiYy00Zjg5LTRhNjEtYTkzNi0zNmQyNDI3MWFmMmIiLCJqdGkiOiJkMWVlNDM2NC05MjU5LTQzNmEtYjY0My04YzdkZDQ3ZjlmOWMiLCJ0eXBlIjoicmVmcmVzaCIsImlhdCI6MTc2MzkxMDg0MywiZXhwIjoxNzY0NTE1NjQzfQ.IlQPZAPbdATeUkoWimGKpKWrSodatNaR7FAfk__OkTA	2025-11-30 22:14:03.072662+07	f
2f6c4759-5784-45f5-97a7-125cb8d317f6	4ace5bbc-4f89-4a61-a936-36d24271af2b	eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI0YWNlNWJiYy00Zjg5LTRhNjEtYTkzNi0zNmQyNDI3MWFmMmIiLCJqdGkiOiIyZjZjNDc1OS01Nzg0LTQ1ZjUtOTdhNy0xMjVjYjhkMzE3ZjYiLCJ0eXBlIjoicmVmcmVzaCIsImlhdCI6MTc2MzkxMDE5OCwiZXhwIjoxNzY0NTE0OTk4fQ.KwLIP523VG2Y3WXVnHn5GSqIFfECIdvgZdmSTA9ljTg	2025-11-30 22:03:18.150246+07	f
074e4529-2ede-42eb-9df5-368fad7cc24c	4ace5bbc-4f89-4a61-a936-36d24271af2b	eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI0YWNlNWJiYy00Zjg5LTRhNjEtYTkzNi0zNmQyNDI3MWFmMmIiLCJqdGkiOiIwNzRlNDUyOS0yZWRlLTQyZWItOWRmNS0zNjhmYWQ3Y2MyNGMiLCJ0eXBlIjoicmVmcmVzaCIsImlhdCI6MTc2MzkxNjc5MCwiZXhwIjoxNzY0NTIxNTkwfQ.4JMOMgjpCnOHJi3IcCmqA4hZ88uUz7gxjhJyb5miRRs	2025-11-30 23:53:10.869422+07	f
a0b6b382-ad66-4a3f-bd46-bd769b4f05a2	5dbe3908-51e7-4a85-91ed-1e5fd4c5a788	eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI1ZGJlMzkwOC01MWU3LTRhODUtOTFlZC0xZTVmZDRjNWE3ODgiLCJqdGkiOiJhMGI2YjM4Mi1hZDY2LTRhM2YtYmQ0Ni1iZDc2OWI0ZjA1YTIiLCJ0eXBlIjoicmVmcmVzaCIsImlhdCI6MTc2MzkxMTEzOSwiZXhwIjoxNzY0NTE1OTM5fQ.3fH0fPnF1pPlvIdyQlrO4eno30caiSBKHjpJwpxwC2M	2025-11-30 22:18:59.882103+07	f
053e01b4-cd01-4334-ac24-b64e4abf4b1b	35941a99-7d6b-4e75-9847-085114cd3389	eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIzNTk0MWE5OS03ZDZiLTRlNzUtOTg0Ny0wODUxMTRjZDMzODkiLCJqdGkiOiIwNTNlMDFiNC1jZDAxLTQzMzQtYWMyNC1iNjRlNGFiZjRiMWIiLCJ0eXBlIjoicmVmcmVzaCIsImlhdCI6MTc2MzkxMTE2NCwiZXhwIjoxNzY0NTE1OTY0fQ.2D1a3GFPUndZXBQn7w8bVq0wIO4SdIoJG9Kuyufeu5k	2025-11-30 22:19:24.374658+07	f
cb9b2555-0171-4d51-a791-98229f3b645c	5dbe3908-51e7-4a85-91ed-1e5fd4c5a788	eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI1ZGJlMzkwOC01MWU3LTRhODUtOTFlZC0xZTVmZDRjNWE3ODgiLCJqdGkiOiJjYjliMjU1NS0wMTcxLTRkNTEtYTc5MS05ODIyOWYzYjY0NWMiLCJ0eXBlIjoicmVmcmVzaCIsImlhdCI6MTc2MzkxNjc5MywiZXhwIjoxNzY0NTIxNTkzfQ.LoofXRCLH20xCN8VnQdZFBFE47ggFTEsk2JkSTMq6L8	2025-11-30 23:53:13.136109+07	f
5c3fb93f-6703-4db9-b2ec-ed51f528b3bf	4ace5bbc-4f89-4a61-a936-36d24271af2b	eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI0YWNlNWJiYy00Zjg5LTRhNjEtYTkzNi0zNmQyNDI3MWFmMmIiLCJqdGkiOiI1YzNmYjkzZi02NzAzLTRkYjktYjJlYy1lZDUxZjUyOGIzYmYiLCJ0eXBlIjoicmVmcmVzaCIsImlhdCI6MTc2MzkxNjMyNSwiZXhwIjoxNzY0NTIxMTI1fQ.JqfGl3LTmUj6o54ZutZ7mg4QxFBst4OKoJ9bkECvRl8	2025-11-30 23:45:25.963426+07	f
c4f00814-e925-475e-8e3c-8f17d04691cb	4ace5bbc-4f89-4a61-a936-36d24271af2b	eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI0YWNlNWJiYy00Zjg5LTRhNjEtYTkzNi0zNmQyNDI3MWFmMmIiLCJqdGkiOiJjNGYwMDgxNC1lOTI1LTQ3NWUtOGUzYy04ZjE3ZDA0NjkxY2IiLCJ0eXBlIjoicmVmcmVzaCIsImlhdCI6MTc2MzkxMjQwMCwiZXhwIjoxNzY0NTE3MjAwfQ.IC9Zj5COS9GZm-7TeApUyEncI5qKZa0UimcmbnBLk9c	2025-11-30 22:40:00.879556+07	f
21ac598d-c763-454a-90d3-416018a175d7	4ace5bbc-4f89-4a61-a936-36d24271af2b	eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI0YWNlNWJiYy00Zjg5LTRhNjEtYTkzNi0zNmQyNDI3MWFmMmIiLCJqdGkiOiIyMWFjNTk4ZC1jNzYzLTQ1NGEtOTBkMy00MTYwMThhMTc1ZDciLCJ0eXBlIjoicmVmcmVzaCIsImlhdCI6MTc2NDAwMzU3MCwiZXhwIjoxNzY0NjA4MzcwfQ.c1XFYI1D-ncvUu1Ru-6ADXKnI1jWOlOScVLLSSBcwJ8	2025-12-01 23:59:30.727797+07	f
c45ef7e3-9314-48ec-9d1d-7e2a69d1f1d4	4ace5bbc-4f89-4a61-a936-36d24271af2b	eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI0YWNlNWJiYy00Zjg5LTRhNjEtYTkzNi0zNmQyNDI3MWFmMmIiLCJqdGkiOiJjNDVlZjdlMy05MzE0LTQ4ZWMtOWQxZC03ZTJhNjlkMWYxZDQiLCJ0eXBlIjoicmVmcmVzaCIsImlhdCI6MTc2MzkxNDE3MSwiZXhwIjoxNzY0NTE4OTcxfQ.s6Nz9tkuKuw09loG7oQXhxgZdu4msKH9dQ111XWd2zE	2025-11-30 23:09:31.865234+07	f
027f4988-7c14-4f70-aef2-8b594c927d41	4ace5bbc-4f89-4a61-a936-36d24271af2b	eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI0YWNlNWJiYy00Zjg5LTRhNjEtYTkzNi0zNmQyNDI3MWFmMmIiLCJqdGkiOiIwMjdmNDk4OC03YzE0LTRmNzAtYWVmMi04YjU5NGM5MjdkNDEiLCJ0eXBlIjoicmVmcmVzaCIsImlhdCI6MTc2MzkxNTA2MywiZXhwIjoxNzY0NTE5ODYzfQ.t0-YA3Tg0uvMGYmX203VVGHOyTnkxmarXc22R_G2Rwc	2025-11-30 23:24:23.463749+07	f
21582d34-74a1-4fe8-8840-41e04fefa7ff	4ace5bbc-4f89-4a61-a936-36d24271af2b	eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI0YWNlNWJiYy00Zjg5LTRhNjEtYTkzNi0zNmQyNDI3MWFmMmIiLCJqdGkiOiIyMTU4MmQzNC03NGExLTRmZTgtODg0MC00MWUwNGZlZmE3ZmYiLCJ0eXBlIjoicmVmcmVzaCIsImlhdCI6MTc2MzkwMjUwNywiZXhwIjoxNzY0NTA3MzA3fQ.k_pyqxgR0Oftqb0EX5b7RMqcc08uyyOclrED3kW5KbI	2025-11-30 19:55:07.911041+07	f
4952f7c4-f2b3-4566-93e9-778e9c36ca60	4ace5bbc-4f89-4a61-a936-36d24271af2b	eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI0YWNlNWJiYy00Zjg5LTRhNjEtYTkzNi0zNmQyNDI3MWFmMmIiLCJqdGkiOiI0OTUyZjdjNC1mMmIzLTQ1NjYtOTNlOS03NzhlOWMzNmNhNjAiLCJ0eXBlIjoicmVmcmVzaCIsImlhdCI6MTc2MzkwNzIwMiwiZXhwIjoxNzY0NTEyMDAyfQ.klYAohiQDbvgqmesyEIhukwAo_VhqwVCAm7lHYwtUGI	2025-11-30 21:13:22.129665+07	f
db9b86cb-9367-435d-93c3-5c53169612f5	4ace5bbc-4f89-4a61-a936-36d24271af2b	eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI0YWNlNWJiYy00Zjg5LTRhNjEtYTkzNi0zNmQyNDI3MWFmMmIiLCJqdGkiOiJkYjliODZjYi05MzY3LTQzNWQtOTNjMy01YzUzMTY5NjEyZjUiLCJ0eXBlIjoicmVmcmVzaCIsImlhdCI6MTc2MzkxNTk4MCwiZXhwIjoxNzY0NTIwNzgwfQ.Ra9HDIn1ab93-ZsE53pF4kpO3my3lQ0oJVkEuRh68JA	2025-11-30 23:39:40.84562+07	f
edef6ceb-2485-44f1-96e5-8f19fbdd9b0a	4ace5bbc-4f89-4a61-a936-36d24271af2b	eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI0YWNlNWJiYy00Zjg5LTRhNjEtYTkzNi0zNmQyNDI3MWFmMmIiLCJqdGkiOiJlZGVmNmNlYi0yNDg1LTQ0ZjEtOTZlNS04ZjE5ZmJkZDliMGEiLCJ0eXBlIjoicmVmcmVzaCIsImlhdCI6MTc2MzkwMzgyMiwiZXhwIjoxNzY0NTA4NjIyfQ.IYjmQnsJaoohZpsCu7Dy6EBuXXA0TpzTbCGW_XlEOZs	2025-11-30 20:17:02.744482+07	f
050272b7-3d86-46a6-a3e7-ee9bfd91e41f	4ace5bbc-4f89-4a61-a936-36d24271af2b	eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI0YWNlNWJiYy00Zjg5LTRhNjEtYTkzNi0zNmQyNDI3MWFmMmIiLCJqdGkiOiIwNTAyNzJiNy0zZDg2LTQ2YTYtYTNlNy1lZTliZmQ5MWU0MWYiLCJ0eXBlIjoicmVmcmVzaCIsImlhdCI6MTc2MzkyODU0MSwiZXhwIjoxNzY0NTMzMzQxfQ.1T0jOIvgS6C41SI0fMqx-Uffa7Rq0uAsr4mRjGCcDlg	2025-12-01 03:09:01.425795+07	f
e0a86ebe-e0c4-476f-8482-a52367c6b035	4ace5bbc-4f89-4a61-a936-36d24271af2b	eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI0YWNlNWJiYy00Zjg5LTRhNjEtYTkzNi0zNmQyNDI3MWFmMmIiLCJqdGkiOiJlMGE4NmViZS1lMGM0LTQ3NmYtODQ4Mi1hNTIzNjdjNmIwMzUiLCJ0eXBlIjoicmVmcmVzaCIsImlhdCI6MTc2MzkwNjU4NCwiZXhwIjoxNzY0NTExMzg0fQ.1lOJz8ZOLOiT5It7uuPnIRQB9_oB-O1QMxP85sHtMEQ	2025-11-30 21:03:04.943627+07	f
9e4ee875-3cd6-4620-9ad2-a2858935b66b	4ace5bbc-4f89-4a61-a936-36d24271af2b	eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI0YWNlNWJiYy00Zjg5LTRhNjEtYTkzNi0zNmQyNDI3MWFmMmIiLCJqdGkiOiI5ZTRlZTg3NS0zY2Q2LTQ2MjAtOWFkMi1hMjg1ODkzNWI2NmIiLCJ0eXBlIjoicmVmcmVzaCIsImlhdCI6MTc2MzkwODQ1NCwiZXhwIjoxNzY0NTEzMjU0fQ.FJ2c2TGo3_J3mbyz3CFU4NiK2ww09GU-JGHs37EYjok	2025-11-30 21:34:14.989056+07	f
bf2b0160-8b14-43b9-bc7c-2b060f02bd4d	1bebabf4-19b5-43d1-8677-c0dcd2bcc928	eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxYmViYWJmNC0xOWI1LTQzZDEtODY3Ny1jMGRjZDJiY2M5MjgiLCJqdGkiOiJiZjJiMDE2MC04YjE0LTQzYjktYmM3Yy0yYjA2MGYwMmJkNGQiLCJ0eXBlIjoicmVmcmVzaCIsImlhdCI6MTc2MzkwODUyNiwiZXhwIjoxNzY0NTEzMzI2fQ.J61hf0GjeoG0SvG-uVfQL9JxhvGzAjw3UUiTPbFLrhU	2025-11-30 21:35:26.293659+07	f
79ce5cf4-101a-49fe-96bf-e07b18b37e97	4ace5bbc-4f89-4a61-a936-36d24271af2b	eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI0YWNlNWJiYy00Zjg5LTRhNjEtYTkzNi0zNmQyNDI3MWFmMmIiLCJqdGkiOiI3OWNlNWNmNC0xMDFhLTQ5ZmUtOTZiZi1lMDdiMThiMzdlOTciLCJ0eXBlIjoicmVmcmVzaCIsImlhdCI6MTc2MzkzNjU5NywiZXhwIjoxNzY0NTQxMzk3fQ.GDTfx8VZZMUljhtx_2EyilOMCWIJeGKpLITBatHcdes	2025-12-01 05:23:17.804012+07	f
57eb9880-5716-4208-841e-11a6b61a8000	5dbe3908-51e7-4a85-91ed-1e5fd4c5a788	eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI1ZGJlMzkwOC01MWU3LTRhODUtOTFlZC0xZTVmZDRjNWE3ODgiLCJqdGkiOiI1N2ViOTg4MC01NzE2LTQyMDgtODQxZS0xMWE2YjYxYTgwMDAiLCJ0eXBlIjoicmVmcmVzaCIsImlhdCI6MTc2MzkyNTUwNSwiZXhwIjoxNzY0NTMwMzA1fQ.09_pVS0Ogu5YuGZdeFJtSHhuNwLjAJ8anqDeOHwRWNs	2025-12-01 02:18:25.720314+07	f
28c670ea-a764-4e4a-a72c-545fcc1f6cbd	5dbe3908-51e7-4a85-91ed-1e5fd4c5a788	eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI1ZGJlMzkwOC01MWU3LTRhODUtOTFlZC0xZTVmZDRjNWE3ODgiLCJqdGkiOiIyOGM2NzBlYS1hNzY0LTRlNGEtYTcyYy01NDVmY2MxZjZjYmQiLCJ0eXBlIjoicmVmcmVzaCIsImlhdCI6MTc2MzkzOTMzNSwiZXhwIjoxNzY0NTQ0MTM1fQ.3oY_zhGtEb7SwY3C-B4WJe5ONbQUi8oA6tkHzJrOmtU	2025-12-01 06:08:55.548961+07	f
5238a564-21d4-4191-93c8-5248c6cee2f3	4ace5bbc-4f89-4a61-a936-36d24271af2b	eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI0YWNlNWJiYy00Zjg5LTRhNjEtYTkzNi0zNmQyNDI3MWFmMmIiLCJqdGkiOiI1MjM4YTU2NC0yMWQ0LTQxOTEtOTNjOC01MjQ4YzZjZWUyZjMiLCJ0eXBlIjoicmVmcmVzaCIsImlhdCI6MTc2MzkzNDU1MywiZXhwIjoxNzY0NTM5MzUzfQ.0rGZAYBRPXhZCcR0qcPKwN5TmfC9YUOchO4ZAg0CAFg	2025-12-01 04:49:13.034768+07	f
39e9dc38-8ef9-46ca-a6a0-5af4e3104ba7	4ace5bbc-4f89-4a61-a936-36d24271af2b	eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI0YWNlNWJiYy00Zjg5LTRhNjEtYTkzNi0zNmQyNDI3MWFmMmIiLCJqdGkiOiIzOWU5ZGMzOC04ZWY5LTQ2Y2EtYTZhMC01YWY0ZTMxMDRiYTciLCJ0eXBlIjoicmVmcmVzaCIsImlhdCI6MTc2MzkxOTYyMiwiZXhwIjoxNzY0NTI0NDIyfQ.BWQTM0FNQ7wDZBA9ndM7opznQZydRc9xvB_5YIs49Ro	2025-12-01 00:40:22.581487+07	f
7c9b335a-4670-4b81-82f7-a9f76d9c2a40	5dbe3908-51e7-4a85-91ed-1e5fd4c5a788	eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI1ZGJlMzkwOC01MWU3LTRhODUtOTFlZC0xZTVmZDRjNWE3ODgiLCJqdGkiOiI3YzliMzM1YS00NjcwLTRiODEtODJmNy1hOWY3NmQ5YzJhNDAiLCJ0eXBlIjoicmVmcmVzaCIsImlhdCI6MTc2MzkyNzg5MSwiZXhwIjoxNzY0NTMyNjkxfQ.uMec4ldt2UsVbCIM27XxuuzKE_LN9o3zbSgQrbdmp0I	2025-12-01 02:58:11.228421+07	f
08e1c8e9-73da-411b-a50b-2187beb593fe	4ace5bbc-4f89-4a61-a936-36d24271af2b	eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI0YWNlNWJiYy00Zjg5LTRhNjEtYTkzNi0zNmQyNDI3MWFmMmIiLCJqdGkiOiIwOGUxYzhlOS03M2RhLTQxMWItYTUwYi0yMTg3YmViNTkzZmUiLCJ0eXBlIjoicmVmcmVzaCIsImlhdCI6MTc2MzkyNjEwMSwiZXhwIjoxNzY0NTMwOTAxfQ.C-MhSoZlMD5HsainZXjSdr_noUNT4m8PDHZUp32k10g	2025-12-01 02:28:21.20534+07	f
e3157a53-0995-4c51-831b-88f3b4b7009a	4ace5bbc-4f89-4a61-a936-36d24271af2b	eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI0YWNlNWJiYy00Zjg5LTRhNjEtYTkzNi0zNmQyNDI3MWFmMmIiLCJqdGkiOiJlMzE1N2E1My0wOTk1LTRjNTEtODMxYi04OGYzYjRiNzAwOWEiLCJ0eXBlIjoicmVmcmVzaCIsImlhdCI6MTc2NDAxMDkzOSwiZXhwIjoxNzY0NjE1NzM5fQ.KQAxWZnVc7HQiQ9uI4_4V4a5Z_Sc1-T10YQhQz_Y1zY	2025-12-02 02:02:19.276631+07	f
7df3fddb-4bc0-456b-a1e6-49af61c90b9a	4ace5bbc-4f89-4a61-a936-36d24271af2b	eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI0YWNlNWJiYy00Zjg5LTRhNjEtYTkzNi0zNmQyNDI3MWFmMmIiLCJqdGkiOiI3ZGYzZmRkYi00YmMwLTQ1NmItYTFlNi00OWFmNjFjOTBiOWEiLCJ0eXBlIjoicmVmcmVzaCIsImlhdCI6MTc2NDAwNDY4OCwiZXhwIjoxNzY0NjA5NDg4fQ.ClAFBvZmMaQ4qR3VSv6cUydcXRJCetuRajIBfTtliho	2025-12-02 00:18:08.353418+07	f
4351d762-9faf-4803-a615-3e674964c641	4ace5bbc-4f89-4a61-a936-36d24271af2b	eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI0YWNlNWJiYy00Zjg5LTRhNjEtYTkzNi0zNmQyNDI3MWFmMmIiLCJqdGkiOiI0MzUxZDc2Mi05ZmFmLTQ4MDMtYTYxNS0zZTY3NDk2NGM2NDEiLCJ0eXBlIjoicmVmcmVzaCIsImlhdCI6MTc2Mzk0MjI3MCwiZXhwIjoxNzY0NTQ3MDcwfQ.iaNB3_V9wYOcFyEEtPF-xYGma0Wc1xnPsp4Vdx_ZHoc	2025-12-01 06:57:50.36314+07	f
d45fe077-e147-43f1-8dc5-75590243a2ff	5dbe3908-51e7-4a85-91ed-1e5fd4c5a788	eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI1ZGJlMzkwOC01MWU3LTRhODUtOTFlZC0xZTVmZDRjNWE3ODgiLCJqdGkiOiJkNDVmZTA3Ny1lMTQ3LTQzZjEtOGRjNS03NTU5MDI0M2EyZmYiLCJ0eXBlIjoicmVmcmVzaCIsImlhdCI6MTc2MzkyNzcxMSwiZXhwIjoxNzY0NTMyNTExfQ.AoiYNAqu8-2k0S2SqRue41PcBN5jOYoxQOQUrzh9IDw	2025-12-01 02:55:11.950745+07	f
de55166c-098c-441e-99ff-44c6934dc697	4ace5bbc-4f89-4a61-a936-36d24271af2b	eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI0YWNlNWJiYy00Zjg5LTRhNjEtYTkzNi0zNmQyNDI3MWFmMmIiLCJqdGkiOiJkZTU1MTY2Yy0wOThjLTQ0MWUtOTlmZi00NGM2OTM0ZGM2OTciLCJ0eXBlIjoicmVmcmVzaCIsImlhdCI6MTc2MzkyOTU2MSwiZXhwIjoxNzY0NTM0MzYxfQ.K4-NsnKQvOPvs_uxloxrcJ1EzkVBeI38QGBj57eOJqg	2025-12-01 03:26:01.77434+07	f
7efab7a6-7216-413b-bdcf-f8fa9f95740f	5dbe3908-51e7-4a85-91ed-1e5fd4c5a788	eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI1ZGJlMzkwOC01MWU3LTRhODUtOTFlZC0xZTVmZDRjNWE3ODgiLCJqdGkiOiI3ZWZhYjdhNi03MjE2LTQxM2ItYmRjZi1mOGZhOWY5NTc0MGYiLCJ0eXBlIjoicmVmcmVzaCIsImlhdCI6MTc2NDAxMjI0MywiZXhwIjoxNzY0NjE3MDQzfQ.BnGZ3xQyw4m6wkrWl-9O_0gEKrt7-QRtXTKdurR22u8	2025-12-02 02:24:02.999446+07	f
843762a4-4adf-461b-b809-22861b301956	5dbe3908-51e7-4a85-91ed-1e5fd4c5a788	eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI1ZGJlMzkwOC01MWU3LTRhODUtOTFlZC0xZTVmZDRjNWE3ODgiLCJqdGkiOiI4NDM3NjJhNC00YWRmLTQ2MWItYjgwOS0yMjg2MWIzMDE5NTYiLCJ0eXBlIjoicmVmcmVzaCIsImlhdCI6MTc2MzkyOTk2NiwiZXhwIjoxNzY0NTM0NzY2fQ.ScyXPLppxFyjo3LEGOGB2mvj2o3xrAK2d5urjWoo5HI	2025-12-01 03:32:46.033702+07	f
f925d984-8fd9-4e22-9d04-8bd12ef572a2	4ace5bbc-4f89-4a61-a936-36d24271af2b	eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI0YWNlNWJiYy00Zjg5LTRhNjEtYTkzNi0zNmQyNDI3MWFmMmIiLCJqdGkiOiJmOTI1ZDk4NC04ZmQ5LTRlMjItOWQwNC04YmQxMmVmNTcyYTIiLCJ0eXBlIjoicmVmcmVzaCIsImlhdCI6MTc2NDAwODU2NywiZXhwIjoxNzY0NjEzMzY3fQ.DJzM7YuUlellgDHtmLuPvzCuDpEEQGYWiIYAR8KSjaM	2025-12-02 01:22:47.134901+07	f
3176bbc8-ea5f-4158-bb53-5667e6062de1	4ace5bbc-4f89-4a61-a936-36d24271af2b	eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI0YWNlNWJiYy00Zjg5LTRhNjEtYTkzNi0zNmQyNDI3MWFmMmIiLCJqdGkiOiIzMTc2YmJjOC1lYTVmLTQxNTgtYmI1My01NjY3ZTYwNjJkZTEiLCJ0eXBlIjoicmVmcmVzaCIsImlhdCI6MTc2NDA0MDM0NiwiZXhwIjoxNzY0NjQ1MTQ2fQ.004I476xGH31H58p-a7KbyLt_eRSstjeYjFrPmj9Dho	2025-12-02 10:12:26.590685+07	f
a5012130-07ce-4cc5-b685-e9417f8b475c	5dbe3908-51e7-4a85-91ed-1e5fd4c5a788	eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI1ZGJlMzkwOC01MWU3LTRhODUtOTFlZC0xZTVmZDRjNWE3ODgiLCJqdGkiOiJhNTAxMjEzMC0wN2NlLTRjYzUtYjY4NS1lOTQxN2Y4YjQ3NWMiLCJ0eXBlIjoicmVmcmVzaCIsImlhdCI6MTc2MzkyOTYwNSwiZXhwIjoxNzY0NTM0NDA1fQ.Iak6tFOeBuJ1bqpBseF99lLQ6kBWM3hdrePdRwixqPI	2025-12-01 03:26:45.075558+07	f
d4592562-0db6-4216-a902-55716ae82a37	5dbe3908-51e7-4a85-91ed-1e5fd4c5a788	eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI1ZGJlMzkwOC01MWU3LTRhODUtOTFlZC0xZTVmZDRjNWE3ODgiLCJqdGkiOiJkNDU5MjU2Mi0wZGI2LTQyMTYtYTkwMi01NTcxNmFlODJhMzciLCJ0eXBlIjoicmVmcmVzaCIsImlhdCI6MTc2MzkxOTYyMiwiZXhwIjoxNzY0NTI0NDIyfQ.K_sooINMq4cZnpLuiHV7enVPRz-I4uT7lpSaE4wFp_s	2025-12-01 00:40:22.576615+07	f
c86a8645-885c-455f-aa93-6f602893e2e8	5dbe3908-51e7-4a85-91ed-1e5fd4c5a788	eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI1ZGJlMzkwOC01MWU3LTRhODUtOTFlZC0xZTVmZDRjNWE3ODgiLCJqdGkiOiJjODZhODY0NS04ODVjLTQ1NWYtYWE5My02ZjYwMjg5M2UyZTgiLCJ0eXBlIjoicmVmcmVzaCIsImlhdCI6MTc2MzkzNDU2NywiZXhwIjoxNzY0NTM5MzY3fQ.sbSc2Ixz10Yu0yaeP7ofqtzdbKeI7ak2V8SKDdM08wk	2025-12-01 04:49:27.828167+07	f
90a6f936-5705-4d1a-8367-64ca90e89db8	5dbe3908-51e7-4a85-91ed-1e5fd4c5a788	eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI1ZGJlMzkwOC01MWU3LTRhODUtOTFlZC0xZTVmZDRjNWE3ODgiLCJqdGkiOiI5MGE2ZjkzNi01NzA1LTRkMWEtODM2Ny02NGNhOTBlODlkYjgiLCJ0eXBlIjoicmVmcmVzaCIsImlhdCI6MTc2MzkyNjc1MywiZXhwIjoxNzY0NTMxNTUzfQ.6K--EkM0-ZTsbzZCYXDHE2Up0PuZnLMibs-VATYlK2I	2025-12-01 02:39:13.916753+07	f
4d06440b-00e1-4d71-a03e-0b4730eeffce	5dbe3908-51e7-4a85-91ed-1e5fd4c5a788	eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI1ZGJlMzkwOC01MWU3LTRhODUtOTFlZC0xZTVmZDRjNWE3ODgiLCJqdGkiOiI0ZDA2NDQwYi0wMGUxLTRkNzEtYTAzZS0wYjQ3MzBlZWZmY2UiLCJ0eXBlIjoicmVmcmVzaCIsImlhdCI6MTc2NDAwODU2OSwiZXhwIjoxNzY0NjEzMzY5fQ.Vadp80A13Af62YRMcqY-BXMqwXPKAePvV_IzuwdvCtA	2025-12-02 01:22:49.037504+07	f
1e8e3b21-3b7f-49e5-becf-8da79a467ac1	4ace5bbc-4f89-4a61-a936-36d24271af2b	eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI0YWNlNWJiYy00Zjg5LTRhNjEtYTkzNi0zNmQyNDI3MWFmMmIiLCJqdGkiOiIxZThlM2IyMS0zYjdmLTQ5ZTUtYmVjZi04ZGE3OWE0NjdhYzEiLCJ0eXBlIjoicmVmcmVzaCIsImlhdCI6MTc2NDAxMjIyNSwiZXhwIjoxNzY0NjE3MDI1fQ.k496gsLaNHCQx8d8K4kxxHStv0R2RDGSCN-j8VM_Tzw	2025-12-02 02:23:45.216483+07	f
8c4fe8a3-1ace-4640-af96-4f15850e9f3f	4ace5bbc-4f89-4a61-a936-36d24271af2b	eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI0YWNlNWJiYy00Zjg5LTRhNjEtYTkzNi0zNmQyNDI3MWFmMmIiLCJqdGkiOiI4YzRmZThhMy0xYWNlLTQ2NDAtYWY5Ni00ZjE1ODUwZTlmM2YiLCJ0eXBlIjoicmVmcmVzaCIsImlhdCI6MTc2Mzk0MzU4NywiZXhwIjoxNzY0NTQ4Mzg3fQ.zLDdXxMOcvZth5P3rX-G_MD6JRyf00VXMKaRaZIjzrM	2025-12-01 07:19:47.19591+07	f
4ccfda03-9863-411d-b290-76260aa4f20e	4ace5bbc-4f89-4a61-a936-36d24271af2b	eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI0YWNlNWJiYy00Zjg5LTRhNjEtYTkzNi0zNmQyNDI3MWFmMmIiLCJqdGkiOiI0Y2NmZGEwMy05ODYzLTQxMWQtYjI5MC03NjI2MGFhNGYyMGUiLCJ0eXBlIjoicmVmcmVzaCIsImlhdCI6MTc2MzkzOTMzNywiZXhwIjoxNzY0NTQ0MTM3fQ.sgupXcejpTMto5elSFr_WixT2KmJC2oalMIhNbB7epM	2025-12-01 06:08:57.144711+07	f
26a86872-2498-4c22-ba84-ccf22124b81d	4ace5bbc-4f89-4a61-a936-36d24271af2b	eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI0YWNlNWJiYy00Zjg5LTRhNjEtYTkzNi0zNmQyNDI3MWFmMmIiLCJqdGkiOiIyNmE4Njg3Mi0yNDk4LTRjMjItYmE4NC1jY2YyMjEyNGI4MWQiLCJ0eXBlIjoicmVmcmVzaCIsImlhdCI6MTc2MzkzNzc4OSwiZXhwIjoxNzY0NTQyNTg5fQ.af9KaF1geq7lQyxAAeNucG-9m9MDnEhaWPqfIlzAn-A	2025-12-01 05:43:09.819251+07	f
5a16ce60-736a-4e29-9ad2-2df47636c8a5	5dbe3908-51e7-4a85-91ed-1e5fd4c5a788	eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI1ZGJlMzkwOC01MWU3LTRhODUtOTFlZC0xZTVmZDRjNWE3ODgiLCJqdGkiOiI1YTE2Y2U2MC03MzZhLTRlMjktOWFkMi0yZGY0NzYzNmM4YTUiLCJ0eXBlIjoicmVmcmVzaCIsImlhdCI6MTc2NDAxMDkxMCwiZXhwIjoxNzY0NjE1NzEwfQ.QjkyA3Lpoix3OR8oz0dLSMrZwFK6VHkgXco1sJsu-9k	2025-12-02 02:01:50.00982+07	f
05592c26-ce71-4ea1-bf43-4eb94bd715cf	4ace5bbc-4f89-4a61-a936-36d24271af2b	eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI0YWNlNWJiYy00Zjg5LTRhNjEtYTkzNi0zNmQyNDI3MWFmMmIiLCJqdGkiOiIwNTU5MmMyNi1jZTcxLTRlYTEtYmY0My00ZWI5NGJkNzE1Y2YiLCJ0eXBlIjoicmVmcmVzaCIsImlhdCI6MTc2NDAwNjY2MCwiZXhwIjoxNzY0NjExNDYwfQ.cyuYpQpUFLwYkJU_BMqyPdCQs5pdKpaRvf482PxaEb8	2025-12-02 00:51:00.492283+07	f
1dee9b6c-b715-4490-8cee-f2a1cac72287	4ace5bbc-4f89-4a61-a936-36d24271af2b	eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI0YWNlNWJiYy00Zjg5LTRhNjEtYTkzNi0zNmQyNDI3MWFmMmIiLCJqdGkiOiIxZGVlOWI2Yy1iNzE1LTQ0OTAtOGNlZS1mMmExY2FjNzIyODciLCJ0eXBlIjoicmVmcmVzaCIsImlhdCI6MTc2MzkyOTk2NCwiZXhwIjoxNzY0NTM0NzY0fQ.E6SWZ0LGCv1uJWLH8CKmlTOIYoX68RUCXzcGaK12mKI	2025-12-01 03:32:44.458376+07	f
d8348afe-409b-4dae-8f8e-8619bbbe7519	4ace5bbc-4f89-4a61-a936-36d24271af2b	eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI0YWNlNWJiYy00Zjg5LTRhNjEtYTkzNi0zNmQyNDI3MWFmMmIiLCJqdGkiOiJkODM0OGFmZS00MDliLTRkYWUtOGY4ZS04NjE5YmJiZTc1MTkiLCJ0eXBlIjoicmVmcmVzaCIsImlhdCI6MTc2NDAwOTU0MywiZXhwIjoxNzY0NjE0MzQzfQ.1Znd0GsM6UCjhNOQboznczfOBTZ4vSCpwfLzqIrCJQY	2025-12-02 01:39:03.13119+07	f
83f16d8d-cf1c-4541-afaf-22e9d8b62099	5dbe3908-51e7-4a85-91ed-1e5fd4c5a788	eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI1ZGJlMzkwOC01MWU3LTRhODUtOTFlZC0xZTVmZDRjNWE3ODgiLCJqdGkiOiI4M2YxNmQ4ZC1jZjFjLTQ1NDEtYWZhZi0yMmU5ZDhiNjIwOTkiLCJ0eXBlIjoicmVmcmVzaCIsImlhdCI6MTc2NDAwOTU0NCwiZXhwIjoxNzY0NjE0MzQ0fQ.K7NAIna1_71NeZxI1stAKSA48tikD7JueLU9W4VRnL0	2025-12-02 01:39:04.181367+07	f
8be0921f-0a5d-4b73-aa37-f20221d2285c	5dbe3908-51e7-4a85-91ed-1e5fd4c5a788	eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI1ZGJlMzkwOC01MWU3LTRhODUtOTFlZC0xZTVmZDRjNWE3ODgiLCJqdGkiOiI4YmUwOTIxZi0wYTVkLTRiNzMtYWEzNy1mMjAyMjFkMjI4NWMiLCJ0eXBlIjoicmVmcmVzaCIsImlhdCI6MTc2NDAxMjgwNSwiZXhwIjoxNzY0NjE3NjA1fQ.FNCbQOg3GkzwAh9gL5OgYQgBXr5RRRWOB_4JKNFpWtk	2025-12-02 02:33:25.347872+07	f
e17e2c1c-a1d2-4cd9-bc57-5c23a2665345	4ace5bbc-4f89-4a61-a936-36d24271af2b	eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI0YWNlNWJiYy00Zjg5LTRhNjEtYTkzNi0zNmQyNDI3MWFmMmIiLCJqdGkiOiJlMTdlMmMxYy1hMWQyLTRjZDktYmM1Ny01YzIzYTI2NjUzNDUiLCJ0eXBlIjoicmVmcmVzaCIsImlhdCI6MTc2MzkzOTIzNiwiZXhwIjoxNzY0NTQ0MDM2fQ.pmQi_SdV12FupNRsqSXrgBD5zE7GJzwbS6WKCiObbTU	2025-12-01 06:07:16.69953+07	f
7b3ca9f1-1462-4e35-8c04-2dd0ed870477	4ace5bbc-4f89-4a61-a936-36d24271af2b	eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI0YWNlNWJiYy00Zjg5LTRhNjEtYTkzNi0zNmQyNDI3MWFmMmIiLCJqdGkiOiI3YjNjYTlmMS0xNDYyLTRlMzUtOGMwNC0yZGQwZWQ4NzA0NzciLCJ0eXBlIjoicmVmcmVzaCIsImlhdCI6MTc2MzkyNTAxMywiZXhwIjoxNzY0NTI5ODEzfQ.noYqDuLHJoOaGu6wtY6nUmYflyxSJ1ZAZnk4jh2nmEw	2025-12-01 02:10:13.452434+07	f
39258b29-3ec9-4df1-82fa-9b7978010c29	4ace5bbc-4f89-4a61-a936-36d24271af2b	eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI0YWNlNWJiYy00Zjg5LTRhNjEtYTkzNi0zNmQyNDI3MWFmMmIiLCJqdGkiOiIzOTI1OGIyOS0zZWM5LTRkZjEtODJmYS05Yjc5NzgwMTBjMjkiLCJ0eXBlIjoicmVmcmVzaCIsImlhdCI6MTc2MzkyNzUyMSwiZXhwIjoxNzY0NTMyMzIxfQ.-ZS-75Jw3JIGmpDk2TvtRmP7yLbUpjp0sAahxgDAaO8	2025-12-01 02:52:01.421915+07	f
3056bc39-2bd8-4df6-8292-583abd432c15	5dbe3908-51e7-4a85-91ed-1e5fd4c5a788	eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI1ZGJlMzkwOC01MWU3LTRhODUtOTFlZC0xZTVmZDRjNWE3ODgiLCJqdGkiOiIzMDU2YmMzOS0yYmQ4LTRkZjYtODI5Mi01ODNhYmQ0MzJjMTUiLCJ0eXBlIjoicmVmcmVzaCIsImlhdCI6MTc2MzkxODI5NiwiZXhwIjoxNzY0NTIzMDk2fQ.wMQyM9A_pt2E2jkBac844Hwxp_K7ha3Ps4ioyDtsQTk	2025-12-01 00:18:16.692009+07	f
9e01d9e4-60a4-4f27-85ba-86bc4b588983	4ace5bbc-4f89-4a61-a936-36d24271af2b	eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI0YWNlNWJiYy00Zjg5LTRhNjEtYTkzNi0zNmQyNDI3MWFmMmIiLCJqdGkiOiI5ZTAxZDllNC02MGE0LTRmMjctODViYS04NmJjNGI1ODg5ODMiLCJ0eXBlIjoicmVmcmVzaCIsImlhdCI6MTc2MzkxODMwMiwiZXhwIjoxNzY0NTIzMTAyfQ.8LNRlMEfTT49_Wtnws1OQUOro3K0MO94IO7xLFLs_QE	2025-12-01 00:18:22.371873+07	f
71496763-d4e5-4247-b0b1-9e3e245f8a20	4ace5bbc-4f89-4a61-a936-36d24271af2b	eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI0YWNlNWJiYy00Zjg5LTRhNjEtYTkzNi0zNmQyNDI3MWFmMmIiLCJqdGkiOiI3MTQ5Njc2My1kNGU1LTQyNDctYjBiMS05ZTNlMjQ1ZjhhMjAiLCJ0eXBlIjoicmVmcmVzaCIsImlhdCI6MTc2Mzk0MTE3OSwiZXhwIjoxNzY0NTQ1OTc5fQ.qfKml8AHAqs4e7Gbgp_eJox5r0dCVkPcWwcfuDpqcfY	2025-12-01 06:39:39.901344+07	f
0e88b2ac-a7ca-405d-972a-316ed19ea1cf	4ace5bbc-4f89-4a61-a936-36d24271af2b	eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI0YWNlNWJiYy00Zjg5LTRhNjEtYTkzNi0zNmQyNDI3MWFmMmIiLCJqdGkiOiIwZTg4YjJhYy1hN2NhLTQwNWQtOTcyYS0zMTZlZDE5ZWExY2YiLCJ0eXBlIjoicmVmcmVzaCIsImlhdCI6MTc2Mzk0NDc0MCwiZXhwIjoxNzY0NTQ5NTQwfQ.PzhSzY_IUKfIL86IWzsp0j1RPgqBaoZqpxggo0vEPoM	2025-12-01 07:39:00.516085+07	f
cc092c89-2332-4f5f-8385-af50864cbc83	4ace5bbc-4f89-4a61-a936-36d24271af2b	eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI0YWNlNWJiYy00Zjg5LTRhNjEtYTkzNi0zNmQyNDI3MWFmMmIiLCJqdGkiOiJjYzA5MmM4OS0yMzMyLTRmNWYtODM4NS1hZjUwODY0Y2JjODMiLCJ0eXBlIjoicmVmcmVzaCIsImlhdCI6MTc2NDAxMjc0OCwiZXhwIjoxNzY0NjE3NTQ4fQ.SQORrcjf05wACqGOJYUc8c7NAz8vlXj6YBTGkJ_qKKg	2025-12-02 02:32:28.966391+07	f
073a653e-b2ab-4dab-8e23-0674ebe9002e	4ace5bbc-4f89-4a61-a936-36d24271af2b	eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI0YWNlNWJiYy00Zjg5LTRhNjEtYTkzNi0zNmQyNDI3MWFmMmIiLCJqdGkiOiIwNzNhNjUzZS1iMmFiLTRkYWItOGUyMy0wNjc0ZWJlOTAwMmUiLCJ0eXBlIjoicmVmcmVzaCIsImlhdCI6MTc2NDAwNzc0MiwiZXhwIjoxNzY0NjEyNTQyfQ.EVpB1EXYloLgJ0-eS5W9szO9Ibzi-H5FLTeE47xP8M0	2025-12-02 01:09:02.766569+07	f
611d3c4f-07fa-4213-ab9c-84a3374837dd	4ace5bbc-4f89-4a61-a936-36d24271af2b	eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI0YWNlNWJiYy00Zjg5LTRhNjEtYTkzNi0zNmQyNDI3MWFmMmIiLCJqdGkiOiI2MTFkM2M0Zi0wN2ZhLTQyMTMtYWI5Yy04NGEzMzc0ODM3ZGQiLCJ0eXBlIjoicmVmcmVzaCIsImlhdCI6MTc2NDAxNzc4NCwiZXhwIjoxNzY0NjIyNTg0fQ.EWVxj9DfsYnreERMOwyr0dJOrhIarnqAwGqvV3Y6sww	2025-12-02 03:56:24.390278+07	f
0d09b170-9b94-4efd-b7e0-a2b0e9094166	5dbe3908-51e7-4a85-91ed-1e5fd4c5a788	eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI1ZGJlMzkwOC01MWU3LTRhODUtOTFlZC0xZTVmZDRjNWE3ODgiLCJqdGkiOiIwZDA5YjE3MC05Yjk0LTRlZmQtYjdlMC1hMmIwZTkwOTQxNjYiLCJ0eXBlIjoicmVmcmVzaCIsImlhdCI6MTc2NDAxNzc5NCwiZXhwIjoxNzY0NjIyNTk0fQ.rFi8y2BYljNWXMqhUBxm_ZhaJPORF4sb3AwQWKMHKj8	2025-12-02 03:56:34.449944+07	f
1c8093b3-d72f-4d5d-bf59-8005697d185f	5dbe3908-51e7-4a85-91ed-1e5fd4c5a788	eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI1ZGJlMzkwOC01MWU3LTRhODUtOTFlZC0xZTVmZDRjNWE3ODgiLCJqdGkiOiIxYzgwOTNiMy1kNzJmLTRkNWQtYmY1OS04MDA1Njk3ZDE4NWYiLCJ0eXBlIjoicmVmcmVzaCIsImlhdCI6MTc2NDAxNTIwMSwiZXhwIjoxNzY0NjIwMDAxfQ.nV3zmnyIyU_b8N75YzGwFjLQb87AUm-I_8HMZRqSjNM	2025-12-02 03:13:21.556376+07	f
c786b632-9c89-4a16-a62b-368181b36d29	4ace5bbc-4f89-4a61-a936-36d24271af2b	eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI0YWNlNWJiYy00Zjg5LTRhNjEtYTkzNi0zNmQyNDI3MWFmMmIiLCJqdGkiOiJjNzg2YjYzMi05Yzg5LTRhMTYtYTYyYi0zNjgxODFiMzZkMjkiLCJ0eXBlIjoicmVmcmVzaCIsImlhdCI6MTc2NDAxNTIxNywiZXhwIjoxNzY0NjIwMDE3fQ.XuDn4Ap5ErRthFocjm5kT9c6dPnHJIru0lkkDeg1fHk	2025-12-02 03:13:37.66228+07	f
\.


--
-- TOC entry 5087 (class 0 OID 16400)
-- Dependencies: 220
-- Data for Name: users; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.users (id, username, email, password_hash, elo_rating, coins, created_at, last_login, display_name, avatar_url, preferences) FROM stdin;
abfb01d6-b86b-44e8-85be-2ef2cca739b9	testuser	test@example.com	$argon2id$v=19$m=65536,t=3,p=4$cQeCHhm1f5eXMPGh+mqG/A$wc1P8Ue7CEQkIrm9JeyUXIAfooWcognfy/5fAz7n/hw	1500	0	2025-11-19 22:09:50.028734+07	2025-11-19 22:10:58.782616+07	\N	\N	\N
66ccc111-bb61-484c-8485-1fe1b564019a	pvp_player1	pvp1@test.com	$argon2id$v=19$m=65536,t=3,p=4$PLNQmSiG4SnB/VOvJNmn3A$Zvfhlqwr+8wdJJT2WTsAqmNfYwlGYStY8MawEx6OHLk	1484	0	2025-11-20 00:30:21.933492+07	\N	\N	\N	\N
a368b186-fa80-4ad6-8b13-1b6d0bfa824d	pvp_player2	pvp2@test.com	$argon2id$v=19$m=65536,t=3,p=4$rPDnsmVITpXc0xOTlB08Kg$Vx4D3zXHyRTYfxdeAKLnHmjo2kKP99XM/717Uo4KO1c	1516	0	2025-11-20 00:30:22.197851+07	\N	\N	\N	\N
35941a99-7d6b-4e75-9847-085114cd3389	lumbu1124	lumbu1124@gmail.com	$argon2id$v=19$m=65536,t=3,p=4$CHe3pSGAZNls1ycZLL3PqQ$mB/sbH8ItCagmoLfOvhEhl2uY7orJzmMFshOLaO5SY0	1500	0	2025-11-23 22:19:24.231059+07	\N	\N	\N	\N
8c76a289-3236-48c2-b681-10ca7747c638	bachusever144	bachusever144@gmail.com	$argon2id$v=19$m=65536,t=3,p=4$zUIm/DsmjintLv0V/O5dxQ$xhDjfdP+CpRSxF/AJfVuvet32XecZgX822xv6MVuntE	2913	0	2025-11-23 22:17:21.551907+07	\N	\N	\N	\N
70def630-fdd8-476e-8103-d2c53b2990ca	test_pvp2	tpvp2@test.com	$argon2id$v=19$m=65536,t=3,p=4$ygXtfioCg8u2qtxVOTC2Xg$GA7W8sSzJhsEQuVC02td1oqRPvkGDBk9iZ7gT8oE7yg	1530	0	2025-11-20 00:34:24.928674+07	2025-11-20 00:35:06.928732+07	\N	\N	\N
973324ab-d5b2-4e71-a94b-58124eb18aae	test_pvp1	tpvp1@test.com	$argon2id$v=19$m=65536,t=3,p=4$v6DKePBrSeWqdTEL1Geq3Q$bcYYaiJq2PgAOXY0zYWcgpINF75NvznwC5ECdXGeJ5s	1470	0	2025-11-20 00:34:24.713297+07	2025-11-20 00:35:06.619731+07	\N	\N	\N
1bebabf4-19b5-43d1-8677-c0dcd2bcc928	admin	admin@admin.com	$argon2id$v=19$m=65536,t=3,p=4$vOyHAdchF/LckRgE23+86w$xRTVSdPQUWdHJaodouthZ+p1uc7Tp/ON3+dX4Y+Hzrg	1500	0	2025-11-20 06:11:42.095187+07	2025-11-21 07:56:24.622201+07	\N	\N	\N
f6439ac7-d46a-49f6-8dd5-fe98ed975726	kandesfx	kandesfx@gmail.com	$argon2id$v=19$m=65536,t=3,p=4$VoEkcP6XZT2+j7lMVfo3qw$UOHT8bT9ovi0wwRHlo2G6Yo00OnWZdeOlQU3ciEmfiQ	1500	0	2025-11-20 01:21:07.373479+07	2025-11-20 01:29:34.784717+07	\N	\N	\N
dbbe6c81-4380-4021-9a87-99df88e22f0b	hai	hai@gmail.com	$argon2id$v=19$m=65536,t=3,p=4$8uSzBmoh85PZQ4+Hbg08zQ$MT+T65x2oND4bp2VCMnChomq7PF/qqSpACoY/PZpuM8	1500	0	2025-11-20 01:38:42.645765+07	\N	\N	\N	\N
5252be7a-5169-4316-8802-c4500826907d	k123	1234@gmail.com	$argon2id$v=19$m=65536,t=3,p=4$d3Em2PiZfmHpjqr44GOyEA$AywXKmLRSOxCeJ5+X53JhxfrbO+rdNpV9EgHmmaRW2g	1500	0	2025-11-20 23:39:27.09761+07	\N	\N	\N	\N
6bcc2e04-e0c7-48cc-ab33-a48f4f4e643d	kaka	kaka123@gmail.com	$argon2id$v=19$m=65536,t=3,p=4$y4jCAavX7xqN8sYxeDZKJQ$Fr6F7nhC8sMnbmGDnqSCPxle4yC01CBFABdgHaFAsM8	1500	0	2025-11-21 00:02:57.353268+07	\N	\N	\N	\N
9be85c2b-0b71-4234-b2e2-cb202499dab3	naocachep	naocachep@huit.edu.vn	$argon2id$v=19$m=65536,t=3,p=4$TNBy2hCpAUyZ5Zvhv3tY2g$VdUUPF1LLY1jdX5Pf1Fdtssh+rXOBjBu5pBxSixDtjA	2611	0	2025-11-23 22:18:07.809024+07	\N	\N	\N	\N
b6b6a32e-fcb0-4b2c-93f8-6c6264a75889	premium_test_user	premium_test@example.com	$argon2id$v=19$m=65536,t=3,p=4$J3xd6BbbxxTziKBABIOqAg$nYukTyfnGuNWV4CscPrJdh0ZhjHFivMhAUQ0OuG9BQI	1603	260	2025-11-19 23:28:23.968629+07	2025-11-20 00:34:50.247402+07	\N	\N	\N
4ace5bbc-4f89-4a61-a936-36d24271af2b	hai123	hai123@gmail.com	$argon2id$v=19$m=65536,t=3,p=4$4A7lqeF1odVgc75ZhmHWlQ$YnGNqtrmNMNAoGO2hDtHcsZlUE1gqhuSi2ePYmfyaS0	2147	0	2025-11-20 02:18:36.156449+07	2025-11-25 10:04:51.761979+07	\N	\N	\N
5dbe3908-51e7-4a85-91ed-1e5fd4c5a788	kientuongvinh311	tuongvinh@gmail.com	$argon2id$v=19$m=65536,t=3,p=4$+GvmketjqtyLyg7r0GEWHw$3cbt7402R45QVTp9bUl0ciZ76Rgy2W2sjxJwJATRqqg	2245	0	2025-11-23 22:18:59.73637+07	2025-11-25 03:52:17.992502+07	\N	\N	\N
\.


--
-- TOC entry 4933 (class 2606 OID 16525)
-- Name: alembic_version alembic_version_pkc; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.alembic_version
    ADD CONSTRAINT alembic_version_pkc PRIMARY KEY (version_num);


--
-- TOC entry 4925 (class 2606 OID 16473)
-- Name: coin_transactions coin_transactions_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.coin_transactions
    ADD CONSTRAINT coin_transactions_pkey PRIMARY KEY (id);


--
-- TOC entry 4923 (class 2606 OID 16451)
-- Name: matches matches_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.matches
    ADD CONSTRAINT matches_pkey PRIMARY KEY (id);


--
-- TOC entry 4931 (class 2606 OID 16514)
-- Name: model_versions model_versions_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.model_versions
    ADD CONSTRAINT model_versions_pkey PRIMARY KEY (id);


--
-- TOC entry 4929 (class 2606 OID 16493)
-- Name: premium_requests premium_requests_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.premium_requests
    ADD CONSTRAINT premium_requests_pkey PRIMARY KEY (id);


--
-- TOC entry 4917 (class 2606 OID 16433)
-- Name: refresh_tokens refresh_tokens_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.refresh_tokens
    ADD CONSTRAINT refresh_tokens_pkey PRIMARY KEY (id);


--
-- TOC entry 4919 (class 2606 OID 16527)
-- Name: refresh_tokens refresh_tokens_token_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.refresh_tokens
    ADD CONSTRAINT refresh_tokens_token_key UNIQUE (token);


--
-- TOC entry 4911 (class 2606 OID 16421)
-- Name: users users_email_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.users
    ADD CONSTRAINT users_email_key UNIQUE (email);


--
-- TOC entry 4913 (class 2606 OID 16417)
-- Name: users users_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.users
    ADD CONSTRAINT users_pkey PRIMARY KEY (id);


--
-- TOC entry 4915 (class 2606 OID 16419)
-- Name: users users_username_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.users
    ADD CONSTRAINT users_username_key UNIQUE (username);


--
-- TOC entry 4926 (class 1259 OID 16518)
-- Name: idx_coin_tx_user; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_coin_tx_user ON public.coin_transactions USING btree (user_id, created_at);


--
-- TOC entry 4920 (class 1259 OID 16557)
-- Name: idx_matches_room_code; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_matches_room_code ON public.matches USING btree (room_code) WHERE (room_code IS NOT NULL);


--
-- TOC entry 4921 (class 1259 OID 16517)
-- Name: idx_matches_started_at; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_matches_started_at ON public.matches USING btree (started_at);


--
-- TOC entry 4927 (class 1259 OID 16519)
-- Name: idx_premium_requests_user; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_premium_requests_user ON public.premium_requests USING btree (user_id);


--
-- TOC entry 4908 (class 1259 OID 16516)
-- Name: idx_users_email; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_users_email ON public.users USING btree (email);


--
-- TOC entry 4909 (class 1259 OID 16515)
-- Name: idx_users_username; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_users_username ON public.users USING btree (username);


--
-- TOC entry 4937 (class 2606 OID 16474)
-- Name: coin_transactions coin_transactions_user_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.coin_transactions
    ADD CONSTRAINT coin_transactions_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.users(id) ON DELETE CASCADE;


--
-- TOC entry 4935 (class 2606 OID 16452)
-- Name: matches matches_black_player_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.matches
    ADD CONSTRAINT matches_black_player_id_fkey FOREIGN KEY (black_player_id) REFERENCES public.users(id);


--
-- TOC entry 4936 (class 2606 OID 16457)
-- Name: matches matches_white_player_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.matches
    ADD CONSTRAINT matches_white_player_id_fkey FOREIGN KEY (white_player_id) REFERENCES public.users(id);


--
-- TOC entry 4938 (class 2606 OID 16499)
-- Name: premium_requests premium_requests_match_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.premium_requests
    ADD CONSTRAINT premium_requests_match_id_fkey FOREIGN KEY (match_id) REFERENCES public.matches(id) ON DELETE CASCADE;


--
-- TOC entry 4939 (class 2606 OID 16494)
-- Name: premium_requests premium_requests_user_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.premium_requests
    ADD CONSTRAINT premium_requests_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.users(id) ON DELETE CASCADE;


--
-- TOC entry 4934 (class 2606 OID 16436)
-- Name: refresh_tokens refresh_tokens_user_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.refresh_tokens
    ADD CONSTRAINT refresh_tokens_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.users(id) ON DELETE CASCADE;


-- Completed on 2025-11-25 20:42:28

--
-- PostgreSQL database dump complete
--

\unrestrict 7NAGMsqxqNBfyLjcgX3ncodbhGcIazBeqRhqEPIRofAFxjvtuSYbTvrOezwJhUG

