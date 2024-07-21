import React from 'react';

import BlogHero from '@/components/BlogHero';
import CustomMDX from '@/components/CustomMDX';

import styles from './postSlug.module.css';
import {loadBlogPost} from '../../helpers/file-helpers';

export async function generateMetadata({ params }) {

  const { frontmatter } = await loadBlogPost(params.postSlug);

  return {
    title: `${frontmatter.title}`,
  };
}

async function BlogPost({ params }) {

  console.log(JSON.stringify(params));

  const { frontmatter,content } = await loadBlogPost(params.postSlug);
  console.log(frontmatter);
  return (
    <article className={styles.wrapper}>
      <BlogHero
        title={frontmatter.title}
        publishedOn={frontmatter.publishedOn}
      />
      <div className={styles.page}>
      <CustomMDX source={content} />
      </div>
    </article>
  );
}

export default BlogPost;
