import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots


plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class DataframesFromJSONL:
    
    """
    
    Used for this specific dataset for now but can be generalized later.
    
    """
    
    def __init__(self, jsonl_file_path):
        self.data = self.load_jsonl_data(jsonl_file_path)
        self.df = self.create_dataframes()
    
    def load_jsonl_data(self, file_path):
        """Load JSONL data into a list of dictionaries"""
        data = []
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                data.append(json.loads(line.strip()))
        return data
    
    def create_dataframes(self):
        """Create structured DataFrames from the nested JSON data"""
        dataframes = {}
        
        # Main candidate info
        candidates = []
        for i, record in enumerate(self.data):
            personal_info = record.get('personal_info', {})
            location = personal_info.get('location', {})
            
            candidate = {
                'candidate_id': i,
                'name': personal_info.get('name', 'Unknown'),
                'email': personal_info.get('email', 'Unknown'),
                'phone': personal_info.get('phone', 'Unknown'),
                'city': location.get('city', 'Unknown'),
                'country': location.get('country', 'Unknown'),
                'remote_preference': location.get('remote_preference', 'Unknown'),
                'summary': personal_info.get('summary', 'Unknown'),
                'linkedin': personal_info.get('linkedin', 'Unknown'),
                'github': personal_info.get('github', 'Unknown')
            }
            candidates.append(candidate)
        dataframes['candidates'] = pd.DataFrame(candidates)
        
        # Experience data
        experiences = []
        for i, record in enumerate(self.data):
            experience_list = record.get('experience', [])
            for exp in experience_list:
                company_info = exp.get('company_info', {})
                dates = exp.get('dates', {})
                tech_env = exp.get('technical_environment', {})
                
                experience = {
                    'candidate_id': i,
                    'company': exp.get('company', 'Unknown'),
                    'title': exp.get('title', 'Unknown'),
                    'level': exp.get('level', 'Unknown'),
                    'employment_type': exp.get('employment_type', 'Unknown'),
                    'start_date': dates.get('start', 'Unknown'),
                    'end_date': dates.get('end', 'Unknown'),
                    'duration': dates.get('duration', 'Unknown'),
                    'industry': company_info.get('industry', 'Unknown'),
                    'company_size': company_info.get('size', 'Unknown'),
                    'technologies': ', '.join(tech_env.get('technologies', [])),
                    'tools': ', '.join(tech_env.get('tools', []))
                }
                experiences.append(experience)
        dataframes['experiences'] = pd.DataFrame(experiences)
        
        # Education data
        educations = []
        for i, record in enumerate(self.data):
            education_list = record.get('education', [])
            for edu in education_list:
                degree = edu.get('degree', {})
                institution = edu.get('institution', {})
                dates = edu.get('dates', {})
                achievements = edu.get('achievements', {})
                
                education = {
                    'candidate_id': i,
                    'degree_level': degree.get('level', 'Unknown'),
                    'field': degree.get('field', 'Unknown'),
                    'institution': institution.get('name', 'Unknown'),
                    'institution_location': institution.get('location', 'Unknown'),
                    'graduation_date': dates.get('expected_graduation', 'Unknown'),
                    'gpa': achievements.get('gpa', None)
                }
                educations.append(education)
        dataframes['educations'] = pd.DataFrame(educations)
        
        # Skills data
        skills = []
        for i, record in enumerate(self.data):
            skills_data = record.get('skills', {})
            technical = skills_data.get('technical', {})
            
            # Programming languages
            prog_langs = technical.get('programming_languages', [])
            for lang in prog_langs:
                if isinstance(lang, dict) and lang.get('name', 'Unknown') != 'Unknown':
                    skills.append({
                        'candidate_id': i,
                        'skill_type': 'programming_language',
                        'skill_name': lang.get('name', 'Unknown'),
                        'skill_level': lang.get('level', 'Unknown')
                    })
            
            # Frameworks
            frameworks = technical.get('frameworks', [])
            for fw in frameworks:
                if isinstance(fw, dict) and fw.get('name', 'Unknown') != 'Unknown':
                    skills.append({
                        'candidate_id': i,
                        'skill_type': 'framework',
                        'skill_name': fw.get('name', 'Unknown'),
                        'skill_level': fw.get('level', 'Unknown')
                    })
            
            # Databases
            databases = technical.get('databases', [])
            for db in databases:
                if isinstance(db, dict) and db.get('name', 'Unknown') != 'Unknown':
                    skills.append({
                        'candidate_id': i,
                        'skill_type': 'database',
                        'skill_name': db.get('name', 'Unknown'),
                        'skill_level': db.get('level', 'Unknown')
                    })
        
        dataframes['skills'] = pd.DataFrame(skills)
        
        return dataframes
    
    

    
    def distribute_candidates_horizontal(self):
        
        
        df_columns = ['name', 'email', 'phone', 'city', 'country', 'remote_preference', 'linkedin', 'github']
        
        n_cols = 3
        n_rows = (len(df_columns) + 1 + n_cols - 1) // n_cols  # +1 for summary
        
        
        subplot_titles = df_columns + ['summary (word count)']
        
        
        fig = make_subplots(
            rows=n_rows, 
            cols=n_cols,
            subplot_titles=subplot_titles,
            specs=[[{"type": "bar"} for _ in range(n_cols)] for _ in range(n_rows)]
        )
        
        
        for i, column in enumerate(df_columns):
            row = (i // n_cols) + 1
            col = (i % n_cols) + 1
            
            column_counts = self.df['candidates'][column].value_counts()
            column_counts = column_counts[column_counts.index != 'Unknown']
            
            if not column_counts.empty:
                top_values = column_counts.head(10)
                
                
                fig.add_trace(
                    go.Bar(
                        y=top_values.index,  # y for horizontal bars
                        x=top_values.values,  # x for horizontal bars
                        orientation='h',      # horizontal orientation
                        name=column,
                        showlegend=False
                    ),
                    row=row, col=col
                )
        
        
        summary_position = len(df_columns)
        summary_row = (summary_position // n_cols) + 1
        summary_col = (summary_position % n_cols) + 1
        

        summaries = self.df['candidates']['summary']
        summaries = summaries[summaries != 'Unknown']  # Filter out 'Unknown'
        
        if not summaries.empty:
            # Count words in each summary
            word_counts = []
            for summary in summaries:
                if pd.notna(summary) and summary != 'Unknown':
                    # Simple word count (split by spaces)
                    word_count = len(str(summary).split())
                    word_counts.append(word_count)
            
            if word_counts:

                bins = [0, 5, 10, 15, 20, 30, 50, float('inf')]
                labels = ['0-5', '6-10', '11-15', '16-20', '21-30', '31-50', '50+']
                
                # Categorize word counts into bins
                word_count_categories = pd.cut(word_counts, bins=bins, labels=labels, right=True)
                category_counts = word_count_categories.value_counts().sort_index()
                
                # Add the word count distribution
                fig.add_trace(
                    go.Bar(
                        y=category_counts.index,
                        x=category_counts.values,
                        orientation='h',
                        name='summary_word_count',
                        showlegend=False,
                        marker_color='lightcoral'  # Different color for summary
                    ),
                    row=summary_row, col=summary_col
                )
        
        fig.update_layout(
            height=400 * n_rows,
            showlegend=False, 
            title_text="Candidates Column Distribution (Horizontal) - Summary by Word Count"
        )
        
        fig.show()
        
    def distribute_skills_horizontal(self):
        
        df_columns = self.df['skills'].columns[1:].tolist()
        
        n_cols = 3
        n_rows = (len(df_columns) + 1 + n_cols - 1) // n_cols
        
        fig = make_subplots(
            rows=n_rows, 
            cols=n_cols,
            specs=[[{"type": "bar"} for _ in range(n_cols)] for _ in range(n_rows)]
        )
        
        
        for i, column in enumerate(df_columns):
            row = (i // n_cols) + 1
            col = (i % n_cols) + 1
            
            column_counts = self.df['skills'][column].value_counts()
            column_counts = column_counts[column_counts.index != 'Unknown']
            
            if not column_counts.empty:
                top_values = column_counts.head(10)
                
                
                fig.add_trace(
                    go.Bar(
                        y=top_values.index,  # y for horizontal bars
                        x=top_values.values,  # x for horizontal bars
                        orientation='h',      # horizontal orientation
                        name=column,
                        showlegend=False
                    ),
                    row=row, col=col
                )
        
        fig.update_layout(
            height=200 * n_rows,
            showlegend=False, 
            title_text="Skills Column Distribution (Horizontal)"
        )
        
        fig.show()
        
    
    def distribute_experiences_horizontal(self):
        
        df_columns = self.df['experiences'].columns[1:].tolist()
        
        n_cols = 3
        n_rows = (len(df_columns) + 1 + n_cols - 1) // n_cols
        
        fig = make_subplots(
            rows=n_rows, 
            cols=n_cols,
            specs=[[{"type": "bar"} for _ in range(n_cols)] for _ in range(n_rows)]
        )
        
        
        for i, column in enumerate(df_columns):
            row = (i // n_cols) + 1
            col = (i % n_cols) + 1
            
            column_counts = self.df['experiences'][column].value_counts()
            column_counts = column_counts[column_counts.index != 'Unknown']
            
            if not column_counts.empty:
                top_values = column_counts.head(10)
                
                
                fig.add_trace(
                    go.Bar(
                        y=top_values.index,  # y for horizontal bars
                        x=top_values.values,  # x for horizontal bars
                        orientation='h',      # horizontal orientation
                        name=column,
                        showlegend=False
                    ),
                    row=row, col=col
                )
        
        fig.update_layout(
            height=400 * n_rows,
            showlegend=False, 
            title_text="Experiences Column Distribution (Horizontal)"
        )
        
        fig.show()
        
    
    
  
    def distribute_educations_horizontal(self):
        
        df_columns = self.df['educations'].columns[1:].tolist()
        
        n_cols = 3
        n_rows = (len(df_columns) + 1 + n_cols - 1) // n_cols
        
        fig = make_subplots(
            rows=n_rows, 
            cols=n_cols,
            specs=[[{"type": "bar"} for _ in range(n_cols)] for _ in range(n_rows)]
        )
        
        
        for i, column in enumerate(df_columns):
            row = (i // n_cols) + 1
            col = (i % n_cols) + 1
            
            column_counts = self.df['educations'][column].value_counts()
            column_counts = column_counts[column_counts.index != 'Unknown']
            
            if not column_counts.empty:
                top_values = column_counts.head(10)
                
                
                fig.add_trace(
                    go.Bar(
                        y=top_values.index,  # y for horizontal bars
                        x=top_values.values,  # x for horizontal bars
                        orientation='h',      # horizontal orientation
                        name=column,
                        showlegend=False
                    ),
                    row=row, col=col
                )
        
        fig.update_layout(
            height=400 * n_rows,
            showlegend=False, 
            title_text="Educations Column Distribution (Horizontal)"
        )
        
        fig.show()
        
    
    def export_summary_report(self, output_file='resume_data_summary.txt'):
        """Export a comprehensive text summary"""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("RESUME DATA ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Total Resumes Analyzed: {len(self.data)}\n\n")
            
            # Geographic analysis
            f.write("GEOGRAPHIC DISTRIBUTION:\n")
            location_counts = self.df['candidates']['city'].value_counts()
            for city, count in location_counts.items():
                f.write(f"  {city}: {count} candidates\n")
            f.write("\n")
            
            # Skills analysis
            f.write("SKILLS ANALYSIS:\n")
            if not self.df['skills'].empty:
                skill_counts = self.df['skills']['skill_name'].value_counts()
                f.write("Top Skills:\n")
                for skill, count in skill_counts.head(15).items():
                    f.write(f"  {skill}: {count} candidates\n")
            f.write("\n")
            
            # Experience analysis
            f.write("EXPERIENCE ANALYSIS:\n")
            if not self.df['experiences'].empty:
                level_counts = self.df['experiences']['level'].value_counts()
                for level, count in level_counts.items():
                    f.write(f"  {level}: {count} positions\n")
            f.write("\n")
            
            # Data quality
            f.write("DATA QUALITY SUMMARY:\n")
            for table_name, df in self.df.items():
                unknown_percentage = 0
                total_cells = 0
                for col in df.columns:
                    if df[col].dtype == 'object' and col != 'candidate_id':
                        unknown_count = df[col].str.contains('Unknown', na=False).sum()
                        null_count = df[col].isnull().sum()
                        total_cells += len(df)
                        unknown_percentage += unknown_count + null_count
                
                if total_cells > 0:
                    completion_rate = 1 - (unknown_percentage / total_cells)
                    f.write(f"  {table_name}: {completion_rate:.1%} data completeness\n")
        
        print(f"Summary report exported to: {output_file}")



