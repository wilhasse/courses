import React from 'react';
import styled from 'styled-components/macro';
import { QUERIES } from '../../constants';

import { MARKET_DATA, SPORTS_STORIES } from '../../data';

import MarketCard from '../MarketCard';
import SectionTitle from '../SectionTitle';
import MiniStory from '../MiniStory';

const SpecialtyStoryGrid = () => {
  return (
    <Wrapper>
      <MarketsSection>
        <SectionTitle
          cornerLink={{
            href: '/markets',
            content: 'Visit Markets data »',
          }}
        >
          Markets
        </SectionTitle>
        <MarketCards>
          {MARKET_DATA.map((data) => (
            <MarketCard key={data.tickerSymbol} {...data} />
          ))}
        </MarketCards>
      </MarketsSection>
      <SportsSection>
        <SectionTitle
          cornerLink={{
            href: '/sports',
            content: 'Visit Sports page »',
          }}
        >
          Sports
        </SectionTitle>
        <SportsStories>
          {SPORTS_STORIES.map((data) => (
            <SportStoryWrapper key={data.id}>
            <MiniStory key={data.id} {...data} />
            </SportStoryWrapper>
          ))}
        </SportsStories>
      </SportsSection>
    </Wrapper>
  );
};

const Wrapper = styled.div`
  display: grid;
  gap: 48px;

  @media ${QUERIES.tabletAndUp} {
    gap: 64px;
    grid-template-columns: minmax(0px, auto);
  }

  @media ${QUERIES.laptopAndUp} {

    grid-template-columns: minmax(0px, auto);
  }
`;

const MarketsSection = styled.section``;

const MarketCards = styled.div`

  @media ${QUERIES.tabletAndUp} {

  display: grid;
  gap: 16px;
  grid-template-columns:
  repeat(auto-fill, minmax(165px,1fr));
  }

  @media ${QUERIES.laptopAndUp} {

    padding-top: 16px;
    margin-right: 16px;
    border-right: 1px solid var(--color-gray-300)};
  }
`;

const SportsSection = styled.section`

`;

const SportStoryWrapper = styled.div` 

  min-width: 220px;
`;

const SportsStories = styled.div`

@media ${QUERIES.tabletAndUp} {

  display: grid;
  gap: 16px;
  grid-template-columns:
  repeat(auto-fill, minmax(165px,1fr));

  @media ${QUERIES.tabletAndUp} {
    display: flex;
    grid-template-columns: revert;
    overflow: auto;
  }
}

`;

export default SpecialtyStoryGrid;
