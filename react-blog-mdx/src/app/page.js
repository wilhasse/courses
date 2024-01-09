import React from 'react';

import BlogSummaryCard from '@/components/BlogSummaryCard';

import styles from './homepage.module.css';

import {getBlogPostList} from '../helpers/file-helpers';
import {BLOG_TITLE} from '../constants';

export const metadata = {
  title: `${BLOG_TITLE}`,
};

async function Home() {

  const blogs = await getBlogPostList();

  return (

    <div className={styles.wrapper}>
      <h1 className={styles.mainHeading}>
        Latest Content:
      </h1>

      {blogs.map(blog => (
        <BlogSummaryCard
          key={blog.slug}
          slug={blog.slug}
          title={blog.title}
          abstract={blog.abstract}
          publishedOn={new Date(blog.publishedOn)} // Assuming 'publishedOn' is a date string
        />
      ))}
     </div>
  );
}

export default Home;
