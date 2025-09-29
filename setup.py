import os
import random
from collections import defaultdict
from typing import Optional

from datasets import load_dataset
from dotenv import load_dotenv
from faker import Faker
from rich import print
from rich.progress import Progress
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from models import Base, User, UserDocument, UserDocumentChunk
from schemas import DocumentItem, IngestionConfig

load_dotenv()

DATASET_ID = "Cohere/wikipedia-22-12-simple-embeddings"
# DATASET_ID = "maloyan/wikipedia-22-12-en-embeddings-all-MiniLM-L6-v2"
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/leo_pgvector")


class DatabaseSetup:
    """Handles database initialization and setup"""

    def __init__(self):
        self.db_url = DATABASE_URL
        self.engine = None
        self.SessionLocal = None

    def connect(self):
        """Establish database connection"""
        print(f"[bold blue]Connecting to database:[/bold blue] {self.db_url.split('@')[-1]}")
        self.engine = create_engine(self.db_url, echo=False)
        self.SessionLocal = sessionmaker(bind=self.engine)
        return self.engine

    def create_tables(self, drop_existing: bool = False):
        """Create database tables"""
        if drop_existing:
            print("[yellow]Dropping existing tables...[/yellow]")
            Base.metadata.drop_all(self.engine)

        print("[bold green]Creating database tables...[/bold green]")
        Base.metadata.create_all(self.engine)
        print("✓ Created database tables")

    def initialize(self, drop_existing: bool = False):
        """Full database initialization"""
        self.connect()
        self.create_tables(drop_existing)
        return self.engine


class DataIngestion:
    """Handles data download and ingestion into the database"""

    def __init__(self, engine, config: Optional[IngestionConfig] = None):
        self.engine = engine
        self.SessionLocal = sessionmaker(bind=engine)
        self.config = config if config else IngestionConfig()
        self.fake = Faker()

        # Categories and subcategories for metadata generation
        self.categories = [
            ("Science", ["Physics", "Chemistry", "Biology", "Astronomy", "Earth Science"]),
            ("Technology", ["Computing", "Engineering", "Electronics", "Software", "Internet"]),
            ("History", ["Ancient", "Medieval", "Modern", "Military", "Cultural"]),
            ("Geography", ["Countries", "Cities", "Landmarks", "Maps", "Climate"]),
            ("Arts", ["Music", "Literature", "Visual Arts", "Film", "Theater"]),
            ("Society", ["Politics", "Economics", "Education", "Religion", "Culture"]),
            ("Biography", ["Scientists", "Politicians", "Artists", "Athletes", "Historical Figures"]),
            ("Medicine", ["Diseases", "Treatments", "Anatomy", "Psychology", "Public Health"]),
        ]

        self.section_types = [
            "introduction",
            "overview",
            "body",
            "details",
            "examples",
            "conclusion",
            "references",
            "see_also",
            "external_links",
        ]
        self.content_types = ["article", "tutorial", "reference", "biography", "list", "timeline"]
        self.difficulty_levels = ["simple", "intermediate", "advanced"]
        self.sentiments = ["positive", "negative", "neutral", "mixed"]

    def create_users(self, session: Session) -> list:
        """Create fake users"""
        print(f"[bold green]Creating {self.config.num_users} users...[/bold green]")

        users = []
        with Progress() as progress:
            task = progress.add_task("Creating users...", total=self.config.num_users)

            for _ in range(self.config.num_users):
                user = User(name=self.fake.name())
                users.append(user)
                session.add(user)
                progress.advance(task)

        session.commit()
        print(f"✓ Created {self.config.num_users} users")
        return users

    def download_dataset(self):
        """Download and process the Cohere Wikipedia dataset"""
        print(f"[bold green]Loading dataset: {self.config.dataset_split}[/bold green]")

        dataset = load_dataset(DATASET_ID, split=self.config.dataset_split)

        print(f"✓ Loaded {len(dataset)} items from dataset")
        return dataset

    def generate_document_metadata(self, title: str, text_chunks: list) -> dict:
        """Generate realistic metadata for a document"""
        category, subcategories = random.choice(self.categories)
        total_text = " ".join([chunk.text for chunk in text_chunks])
        word_count = len(total_text.split())

        return {
            "category": category,
            "subcategory": random.choice(subcategories) if subcategories else None,
            "difficulty_level": random.choice(self.difficulty_levels),
            "content_type": random.choice(self.content_types),
            "quality_score": round(random.uniform(3.0, 10.0), 1),
            "last_updated": self.fake.date_time_between(start_date="-2years", end_date="now").isoformat(),
            "editor_count": random.randint(1, 500),
            "reference_count": random.randint(0, 200),
            "word_count": word_count,
            "image_count": random.randint(0, 50),
            "external_links": random.randint(0, 100),
            "popularity_rank": random.randint(1, 100000),
            "is_featured": random.random() < 0.05,  # 5% chance of being featured
            "language": "en",
            "tags": self.fake.words(nb=random.randint(3, 8), unique=True),
        }

    def generate_chunk_metadata(self, text: str, position_in_doc: int, total_chunks: int) -> dict:
        """Generate realistic metadata for a chunk"""
        word_count = len(text.split())
        char_count = len(text)
        sentence_count = text.count(".") + text.count("!") + text.count("?")

        # Determine position
        if position_in_doc == 0:
            position = "beginning"
            section_type = "introduction"
        elif position_in_doc >= total_chunks - 2:
            position = "end"
            section_type = random.choice(["conclusion", "references", "see_also"])
        else:
            position = "middle"
            section_type = random.choice(["body", "details", "examples"])

        # Detect content features
        has_code = any(marker in text for marker in ["```", "def ", "function", "class ", "import "])
        has_math = any(marker in text for marker in ["=", "+", "-", "*", "/", "equation", "formula"])
        has_list = any(marker in text for marker in ["•", "1.", "2.", "- ", "* "])
        has_table = "|" in text and text.count("|") > 4

        return {
            "section_type": section_type,
            "position": position,
            "word_count": word_count,
            "char_count": char_count,
            "sentence_count": sentence_count,
            "has_code": has_code,
            "has_math": has_math,
            "has_list": has_list,
            "has_table": has_table,
            "language_detected": "en",
            "sentiment": random.choice(self.sentiments),
            "complexity_score": round(random.uniform(1.0, 10.0), 1),
            "readability_score": round(random.uniform(20.0, 80.0), 1),
            "technical_density": round(random.uniform(0.0, 0.5), 2),
            "named_entities": self.fake.words(nb=random.randint(0, 5), unique=True),
            "key_phrases": self.fake.words(nb=random.randint(2, 6), unique=True),
        }

    def process_documents(self, dataset):
        """Process dataset into document groups"""
        print("[bold green]Processing documents...[/bold green]")

        documents_by_url = defaultdict(list)

        with Progress() as progress:
            task = progress.add_task("Processing items...", total=len(dataset))

            for item in dataset:
                # Filter to only include fields defined in DocumentItem
                allowed_keys = DocumentItem.model_fields.keys()
                filtered_data = {k: v for k, v in item.items() if k in allowed_keys}
                doc_item = DocumentItem(**filtered_data)
                documents_by_url[doc_item.url].append(doc_item)
                progress.advance(task)

        print(f"✓ Found {len(documents_by_url)} unique documents")
        return documents_by_url

    def ingest_documents(self, session: Session, users: list, documents_by_url: dict):
        """Ingest documents and chunks into the database"""
        print("[bold green]Ingesting documents and chunks...[/bold green]")

        document_urls = list(documents_by_url.keys())
        total_docs = 0
        total_chunks = 0

        # Shuffle documents for random distribution
        random.shuffle(document_urls)

        # Calculate documents per user for even distribution
        docs_per_user = len(document_urls) // len(users)
        remaining_docs = len(document_urls) % len(users)

        # Track which documents have been assigned
        doc_index = 0

        with Progress() as progress:
            task = progress.add_task("Assigning documents to users...", total=len(users))

            for user_idx, user in enumerate(users):
                # Determine number of documents for this user
                # Add one extra doc to first 'remaining_docs' users to distribute remainder
                num_docs_for_user = docs_per_user + (1 if user_idx < remaining_docs else 0)

                # Don't exceed max_documents_per_user limit
                num_docs_for_user = min(num_docs_for_user, self.config.max_documents_per_user)

                # Assign the next batch of documents to this user
                user_doc_urls = document_urls[doc_index : doc_index + num_docs_for_user]
                doc_index += num_docs_for_user

                # If we've run out of documents, stop
                if not user_doc_urls:
                    progress.advance(task)
                    continue

                for url in user_doc_urls:
                    chunks = documents_by_url[url]
                    first_chunk = chunks[0]

                    # Generate document metadata
                    doc_metadata = self.generate_document_metadata(first_chunk.title, chunks)

                    # Create document
                    user_doc = UserDocument(
                        user_id=user.id,
                        wiki_id=first_chunk.wiki_id,
                        url=first_chunk.url,
                        title=first_chunk.title,
                        views=first_chunk.views,
                        langs=first_chunk.langs,
                        meta=doc_metadata,
                    )
                    session.add(user_doc)
                    session.flush()  # Get the ID
                    total_docs += 1

                    # Create chunks
                    for idx, chunk in enumerate(chunks):
                        # Generate chunk metadata
                        chunk_metadata = self.generate_chunk_metadata(chunk.text, idx, len(chunks))

                        doc_chunk = UserDocumentChunk(
                            user_document_id=user_doc.id,
                            user_id=user.id,  # Set user_id directly for performance
                            paragraph_id=chunk.paragraph_id,
                            text=chunk.text,
                            embedding=chunk.emb,
                            meta=chunk_metadata,
                        )
                        session.add(doc_chunk)
                        total_chunks += 1

                # Commit in batches
                if (users.index(user) + 1) % self.config.batch_size == 0:
                    session.commit()

                progress.advance(task)

        session.commit()
        print(f"✓ Ingested {total_docs} documents with {total_chunks} chunks")

        return total_docs, total_chunks

    def get_statistics(self, session: Session) -> dict:
        """Get database statistics"""
        stats = {
            "total_users": session.query(User).count(),
            "total_documents": session.query(UserDocument).count(),
            "total_chunks": session.query(UserDocumentChunk).count(),
        }

        if stats["total_users"] > 0:
            stats["avg_docs_per_user"] = stats["total_documents"] / stats["total_users"]
        else:
            stats["avg_docs_per_user"] = 0

        return stats

    def run(self):
        """Run the complete data ingestion process"""
        session = self.SessionLocal()

        try:
            # Create users
            users = self.create_users(session)

            # Download and process dataset
            dataset = self.download_dataset()
            documents_by_url = self.process_documents(dataset)

            # Ingest documents
            self.ingest_documents(session, users, documents_by_url)

            # Get statistics
            stats = self.get_statistics(session)

            print("\n[bold blue]Database Statistics:[/bold blue]")
            for key, value in stats.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.2f}")
                else:
                    print(f"  {key}: {value}")

            return stats

        finally:
            session.close()


def main():
    """Main setup function"""
    import argparse

    parser = argparse.ArgumentParser(description="Setup leo-pgvector database")
    parser.add_argument("--drop", action="store_true", help="Drop existing tables")
    parser.add_argument("--skip-data", action="store_true", help="Skip data ingestion")
    parser.add_argument("--users", type=int, default=1000, help="Number of users to create")
    parser.add_argument("--max-docs", type=int, default=10, help="Max documents per user")
    parser.add_argument("--dataset-split", default="train[:5000]", help="Dataset split to load")

    args = parser.parse_args()

    try:
        # Database setup
        db_setup = DatabaseSetup()
        engine = db_setup.initialize(drop_existing=args.drop)

        if not args.skip_data:
            # Data ingestion
            config = IngestionConfig(
                num_users=args.users,
                max_documents_per_user=args.max_docs,
                dataset_split=args.dataset_split,
            )
            ingestion = DataIngestion(engine, config)
            ingestion.run()

        print("\n[bold green]✓ Setup completed successfully![/bold green]")

    except Exception as e:
        print(f"\n[bold red]Error during setup:[/bold red] {e}")
        print("\n[yellow]Make sure PostgreSQL is running and accessible.[/yellow]")
        print("You can set DATABASE_URL environment variable or use default:")
        print("  postgresql://postgres:postgres@localhost:5432/leo_pgvector")
        raise


if __name__ == "__main__":
    main()
